# Import packages
import mne, os, warnings
import pandas as pd
import numpy as np
from autoreject import get_rejection_threshold, AutoReject
from matplotlib import pyplot as plt 
from mne.preprocessing import ICA

from . import PylineData

## Issues Log: Need to check figure plotting saves with correct name and location. 

class PylinePreprocessing:
    """
    Class for performing standard EEG preprocessing steps, including:
    - Signal Space Projection (SSP) for artifact removal
    - Independent Component Analysis (ICA)
    - Filtering (band-pass, notch)
    - Resampling
    - Current Source Density (CSD) transformation
    - Cropping based on events

    Methods should be applied to a PylineData object containing raw EEG data.
    """

    def __init__(self):
        """Initialize PylinePreprocessing object."""
        pass

    def ssp(self, data, params, verbose=False): 
        """
        Apply Signal Space Projection (SSP) for EOG artifact removal.

        Parameters:
        - data (PylineData): EEG data container.
        - params (PylineParameters): Parameters including EOG channels and referencing.
        - verbose (bool): Print detailed progress if True.
        """
        if verbose: print('---\nAPPLYING SSP FOR EOG-REMOVAL\n')

        data_copy = data.copy()

        # Compute SSP projections from EOG channels
        eog_projs, _ = mne.preprocessing.compute_proj_eog(
            data_copy, n_grad=0, n_mag=0, n_eeg=1,
            reject=None, no_proj=True,
            ch_name=params.eog_channels,
            verbose=verbose
        )

        # Apply projections to raw data
        data_copy.add_proj(eog_projs, remove_existing=False)
        data_copy.apply_proj()

        # Drop EOG channels after projection
        if params.reference == 'mastoids':
            eog_channels = [x for x in params.eog_channels if x not in ['EXG1', 'EXG2']]
            data_copy.load_data().drop_channels(eog_channels)
        else:
            data_copy.load_data().drop_channels(params.eog_channels)
        
        return data_copy

    def ica(self, data, n_components=32, method='fastica', plot=False, autoselect=False):
        """
        Apply ICA to EEG data for artifact rejection.

        Parameters:
        - data (PylineData): EEG data container.
        - n_components (int): Number of ICA components.
        - method (str): ICA method, e.g., 'fastica'.
        - plot (bool): Plot components and sources if True.
        - autoselect (bool): Automatically find and exclude EOG components if True.

        Returns:
        - data (PylineData): Updated with fitted ICA object.
        - ica: ICA object
        """
        # Create a filtered copy for ICA fitting
        data_copy = data.copy()
        filt_raw = data.copy().filter(l_freq=1.0, h_freq=30)

        # Fit ICA model
        ica = ICA(n_components=n_components, method=method, random_state=23)
        ica.fit(filt_raw)

        if plot:
            ica.plot_components()
            ica.plot_sources(data, show_scrollbars=False)

        # Auto-exclude EOG components if specified
        if autoselect:
            eog_indices = ica.find_bads_eog(data_copy)
            ica.exclude = eog_indices[0]
            ica.apply(data_copy)

        return ica, data_copy

    def resample(self, data, events, sfreq=None):
        """
        Resample EEG data to a new sampling frequency.

        Parameters:
        - data (PylineData): EEG data container.
        - events (ndarray): MNE-formatted events array.
        - sfreq (int): Desired sampling frequency.

        Returns:
        - resampled_data (Raw): Resampled raw EEG data.
        """
        if sfreq is None or not isinstance(sfreq, int):  
            raise ValueError("Sampling frequency must be specified as integer")

        # Resample and update event timings
        resampled_data, updated_events = data.copy().resample(sfreq=sfreq, events=events)

        return resampled_data, updated_events

    def filter(self, data, params, plot=False, savefig=False, verbose=False):
        """
        Apply band-pass and optional notch filtering to EEG data.

        Parameters:
        - data (Raw): Raw EEG data.
        - params (PylineParameters): Parameters containing filter settings.
        - plot (bool): Plot filter response.
        - savefig (bool): Save filter response plot.
        - verbose (bool): Show MNE messages.

        Returns:
        - filtered_data (Raw): Filtered EEG data.
        """
        if data is None:  
            raise ValueError("No data file saved in data object passed")

        print('---\nAPPLYING FILTER\n')
        filtered_data = data.copy().filter(**params.filter_design, verbose=verbose)

        if plot:
            temp_params = params.filter_design.copy()
            temp_params.pop('n_jobs', None)
            temp_params.pop('pad', None)
            filter_params = mne.filter.create_filter(
                data.get_data(), data.info['sfreq'], **temp_params
            )

            freq_ideal = [0, params.filter_design['l_freq'], params.filter_design['l_freq'],
                          params.filter_design['h_freq'], params.filter_design['h_freq'],
                          data.info['sfreq'] / 2]
            gain_ideal = [0, 0, 1, 1, 0, 0]

            fig, axs = plt.subplots(nrows=3, figsize=(8, 8), layout='tight', dpi=100)
            mne.viz.misc.plot_filter(filter_params, data.info['sfreq'],
                                     freq=freq_ideal, gain=gain_ideal, fscale='log',
                                     flim=(0.01, 80), dlim=(0, 6), axes=axs, show=False)
            if savefig:
                plt.savefig(fname='Data/filter_design.png', dpi=300)
            plt.show()

        # Optional notch filtering for powerline noise
        if params.line_noise is not None:
            if verbose: print('---\nAPPLYING NOTCH FILTER\n')
            filtered_data = filtered_data.notch_filter([params.line_noise])

        return filtered_data

    def csd(self, data):
        """
        Apply Current Source Density (CSD) transformation to EEG data.

        Parameters:
        - data (Raw): EEG data (Raw).

        Returns:
        - None (updates data.csd in place)
        """
        csd_data = mne.preprocessing.compute_current_source_density(data)
        print("Updating data object") 

        return csd_data

    def crop(self, data, events, params): # 15.05.2025 - Needs adjustment to accept any mne data object. 
        """
        Crop EEG data based on detected event markers.

        Parameters:
        - data (MNE-object): EEG data.
        - events (numpy array): event array to drop around (for resting state)
        - params (PylineParameters): Parameters including stimulus channel.

        Returns:
        - cropped_raw (Raw): Cropped EEG data.
        """
        tminmax = None

        if len(events) >= 3:
            tminmax = [events[0][0] / data.info['sfreq'], events[-1][0] / data.info['sfreq']]
            if len(events) > 3:
                warnings.warn('\nMore than 3 event points found for {}\n'.format(data.filenames))
        elif len(events) in [1, 2]:
            warnings.warn('\nOnly 1 or 2 event point(s) found for {}\n'.format(data.filenames))
            if events[0][0] > 100000:
                tminmax = [0, events[0][0] / data.info['sfreq']]
            else:
                tminmax = [events[0][0] / data.info['sfreq'], None]
        else:
            warnings.warn('\nNO event points found for {}\n'.format(data.filenames))

        # Apply cropping
        if tminmax is not None:
            cropped_raw = data.copy().crop(tmin=tminmax[0], tmax=tminmax[1])
            print(('Event markers are following:\n{}\nStarting point: {} s\nEnding point: {} s\n'
                   'Total duration: {} s').format(events, tminmax[0], tminmax[1], tminmax[1] - tminmax[0]))

            if not (230 <= (tminmax[1] - tminmax[0]) <= 250):
                warnings.warn('\nRaw signal length is not between 230-250s for {}\n'.format(data.filenames))
        else:
            print('Signal NOT cropped.')
            cropped_raw = data

        # Drop stimulus channel from cropped data
        cropped_raw = cropped_raw.drop_channels(params.stimulus_channel)
        return cropped_raw

    
    # 14.05.2025 - Re-structered pyline to choice processing option based on tasktype stored in data. Also, only ICA/SSP, re-sampling and filtering done.

    def pyline(self, data, params, manage, analysis, tasktype, ssp_projection = True, sfreq=None, plot=False, savefig=False, savecsv=False,verbose=False):
        """
        Main entry point for preprocessing EEG data.

        Parameters:
        - data (list): List of dictionaries with subject/session/task EEG data paths and metadata.
        - params (object): Configuration object with keys such as `ica`, `event_dict`, and preprocessing parameters.
        - manage (object): Handles paths and data loading (e.g., manage.load_eeg()).
        - analysis (object): Provides export and evoked-related methods.
        - tasktype (str): 'resting' or 'task'.
        - ssp_projection (bool): If True, apply SSP projection (if ICA not used).
        - sfreq (int or None): Desired sampling frequency (e.g., 250 Hz).
        - plot (bool): Whether to display plots.
        - savefig (bool): Whether to save figures.
        - savecsv (bool): Whether to save .csv outputs.
        - verbose (bool): Print progress and warnings.

        Returns:
            - None
        """
        # Validate input types
        if not isinstance(params, object):
            raise TypeError("params must be an object")
        if not isinstance(params.ica, dict):
            raise TypeError("params.ica must be a dictionary")
        if tasktype is None :
            raise TypeError("You must specify either resting or task when running pyline")
        if not isinstance(ssp_projection, bool):
            raise TypeError("ssp_projection must be a boolean")
        if sfreq is not None and not isinstance(sfreq, int):
            raise TypeError("sfreq must be an integer or none")
        if not isinstance(plot, bool):
            raise TypeError("plot must be a boolean")
        if not isinstance(savefig, bool):
            raise TypeError("savefig must be a boolean")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        # Dictionary to map condition types to methods
        condition_specific_code = {
            "resting": self._pyline_resting,
            "task": self._pyline_task,
            # Add more conditions as needed
        }

        # Verify that the tasktype is valid and call the appropriate method
        if tasktype not in condition_specific_code:
            raise ValueError(f"Unsupported condition type: {tasktype}. Add separate function to object EEGprocessing code.")

        # Call the condition-specific method with all relevant parameters
        return condition_specific_code[tasktype](data, params, manage, analysis, tasktype, ssp_projection, sfreq, plot, savefig, savecsv, verbose)

    def _pyline_resting(self, data, params, manage, analysis, tasktype, ssp_projection, sfreq, plot, savefig, savecsv, verbose):
        """
        Resting-state EEG preprocessing pipeline.
        Steps: Load → Crop → ICA/SSP → Filter → Resample → Epoch → Reject → Export
        """
        for entry in data:

            # Set filenames, output directory, and create data object passing params, filename and task identifiers.
            filename = f"{entry['subject']}_{entry['session']}"
            print(f"Running for {filename}")

            output_path = manage.clean_folder / entry['subject'] /entry['session'] / entry['modality']
            data = PylineData(manage.load_eeg(entry['data'], params), filename, tasktype=tasktype)
            paradigm = entry['task']
            params.picks = mne.pick_types(data.raw.info, eeg=True, stim=False) # add picks to params

            # Find events, crop data, apply either ICA or SSP to remove major (repetitive) artifacts, then resample and/or filter
            if params.ica['use']:
                events = self.events_finder(data.raw, params)
                data.events = events

                try:
                    data.raw = self.crop(data.raw, params)
                except Exception as e:
                    print(f"Warning: Error cropping resting data: {e}. If this is a preprocessed file then cropping has potentially already been done.")

                ica_kwargs = {k: v for k, v in params.ica.items() if k != 'use'}
                data.raw = self.ica(data.raw, **ica_kwargs) # Automated ICA rejection applied in arguments
                
            elif ssp_projection:
                events = self.events_finder(data.raw, params)
                data.events = events

                try:
                    data.raw = self.crop(data.raw, params)
                except Exception as e:
                    print(f"Warning: Error cropping resting data: {e}. If this is a preprocessed file then cropping has potentially already been done.")

                data.raw = self.ssp(data.raw, params, verbose=verbose)

            if sfreq == None:
                data.filtered = self.filter(data.raw, params, plot, savefig, verbose)

            else:
                data.resampled, data.events = self.resample(data.raw, data.events, sfreq=sfreq) # resample returns a tuple with resampled data and updated events.

                data.filtered = self.filter(data.resampled, params, plot, savefig, verbose)
                    
            # 14.05.2025 - Potentially remove this and only have the artifact removal, filter, and resample.
            epochs = self.create_epochs(data.filtered, params, tasktype='resting', epo_duration=5)

            ar_epochs = self.reject_auto(epochs)

            analysis.export(ar_epochs, type='fif', outputdir=output_path, filename=f"{data.filename}_{paradigm}")

        return

    # 14.05.2025 - sfreq added but not used. If you want to downsample, refer to the MNE-tutorials on the best approach.
    def _pyline_task(self, data, params, manage, analysis, tasktype, ssp_projection, sfreq, plot, savefig, savecsv, verbose):
        """
        Task-based EEG preprocessing pipeline for paradigms like AX-CPT or AO.
        Includes event splitting, epoching by condition, artifact rejection, and evoked response generation.
        """
        # 14.05.2025 - Removed AX-cue, AB-cue and BX-cue
        if data[0]['task'] == 'AXCPT':
            processing_data = pd.DataFrame(columns=['Subject', 'Group', 'Timepoint', 'Task', 'Time', "AX-target", "AB-target", "BX-target", "AX-target_ar", "AB-target_ar", "BX-target_ar"]) 
        elif data[0]['task'] == 'AO':
            processing_data = pd.DataFrame(columns=['Subject', 'Group', 'Timepoint', 'Task', 'Time', "Standard", "Target", "Standard_ar", "Target_ar"]) 

        trial_outcomes = pd.DataFrame()

        counter = 0

        for entry in data:

            # Set filenames, output directory, and create data object passing params, filename and task identifiers.
            filename = f"{entry['subject']}_{entry['session']}"
            print(f"Running for {filename}")

            output_path = manage.clean_folder / entry['subject'] /entry['session'] / entry['modality']
            data = PylineData(manage.load_eeg(entry['data'], params), filename, tasktype=tasktype)
            paradigm = entry['task']
            params.picks = mne.pick_types(data.raw.info, eeg=True, stim=False) # add picks to params

            processing_data.loc[entry['subject']] = {'Subject': entry['subject'], 'Group': entry['group'], 'Timepoint': entry['session'], 'Task': entry['task'], 'Time': int(data.raw.info['meas_date'].strftime('%H%M'))}


            # Code specific to task data
            if params.ica['use']:
                events = self.events_finder(data.raw, params)
                data.events = events

                ica_kwargs = {k: v for k, v in params.ica.items() if k != 'use'}
                data.raw = self.ica(data, **ica_kwargs) # Automated ICA rejection applied in arguments
            
            elif ssp_projection:
                events = self.events_finder(data.raw, params)
                data.events = events

                data.raw = self.ssp(data.raw, params, verbose=verbose)
            
            data.filtered = self.filter(data.raw, params, plot, savefig, verbose)

            # Split the events into the respective categories for each task
            AX_cue, AX_target, AB_cue, AB_target, BX_cue, BX_target, background_AX, background_A = self.split_events(data.events, params, task="AXCPT")

            # Create event dictionary for creating epochs (modify to include those of interest)
            # 14.05.2025 - Need to either add in AO event dict or remove AO option from this code for upload
            event_dict = {
            "AX-target": [AX_target,{'AX-target': 2}],
            "AB-target": [AB_target, {'A*-target': 8}],
            "BX-target": [BX_target,{'BX-target': 6}]
            }
            
            epoch_dict = {} # Initialize an empty dictionary to store epoch variables
            
            # Epoch around events of interest for each task, passing the data.filtered 
            for key, value in event_dict.items(): # Iterate over the dictionary items and create epochs
                epochs = self.create_epochs(data.filtered, params, tasktype='task', 
                                                events = value[0], event_id = value[1],
                                                title=f"{key.capitalize()} Epochs (GFP without AR)",
                                                epo_duration=None,
                                                plot=False)
                epoch_dict[key] = epochs

                # Add number of epochs to processing dataframe 
                processing_data.loc[entry['subject'], key] = len(epochs)

            # # Process epochs with autoreject, define dictionary of processed epochs for creating evoked objects
            ar_dict = {}

            for key, value in epoch_dict.items():
                # Perform autoreject on each epoch object in dictionary
                ar = self.reject_auto(value, plot=False)

                # Update dictionary with processed epochs (ar is a tuple, so must index variable)
                ar_dict[f"{key}_ar"] = ar[0]

                # Add number of cleaned epochs to processing dataframe 
                processing_data.loc[entry['subject'], f"{key}_ar"] = len(epochs)

                # Save reject log (ar is a tuple, so must index variable)
                ar[1].save(fname=output_path/f"{filename}_{key}_clean_{paradigm}_autorejectlog.npz", overwrite=True)

            # Determine the lowest number of epochs across conditions so epochs can be made the same. This makes the SNR comparable across trials.
            epoch_lens = []

            for key in ar_dict.keys():
                epoch_lens.append(len(ar_dict[key].events))

            minimum_epochs = min(epoch_lens)

            # Add minimum number epochs to processing dataframe for each participant 
            processing_data.loc[entry['subject'], 'min_epochs_post_ar'] = minimum_epochs

            # Use minimum to randomly sample epochs. Export .fif and numpy array for analysis.alpha_reactivity
            for key, value in ar_dict.items():
                #  Use this if you want to equalise across trials.
                # if len(value) > minimum_epochs:
                #     ar_dict[key] = self.select_random_epochs(value, minimum_epochs) 

                analysis.export(ar_dict[key], type='fif', outputdir=output_path, filename=f"{filename}_{key}_clean_{paradigm}")

            # Create evoked objects for each trial type. Export for later ERP analysis.
            for key, value in ar_dict.items():
                evoked = analysis.create_evoked(value)
                analysis.export(evoked, type='fif', outputdir=output_path, filename=f"{filename}_{key}_evoked_{paradigm}")

            # Now, extract relevant processing and task-related data.
            trial_outcomes = analysis.extract_trial_data(trial_outcomes, data, entry, params, task = 'AXCPT')

            counter += 1
            
        if savecsv:
            analysis.export(trial_outcomes, type = 'csv', outputdir=manage.analysis_folder, filename=f'{paradigm}_psychomotor')
            analysis.export(processing_data, type = 'csv', outputdir=manage.analysis_folder, filename=f'{paradigm}_processing_epoch_data') 

        return

    def events_finder(self, data, params, plot = False, task = None):
        # Issue: Needs to not be set as data.raw so user isn't constrained.
        """
        Find events in raw EEG data.

        Parameters:
        - raw_data (mne.io.Raw): Pyline Data object

        Returns:
        - events (ndarray): Array containing event onsets.
        
        """
        events = mne.find_events(data, stim_channel=params.stimulus_channel, consecutive=False, output='onset')

        if plot:
            if task is None:
                raise ValueError('tasktype must be specified to plot events')
            fig = mne.viz.plot_events(
            events, event_id=params.event_dict[task], sfreq=data.info["sfreq"], first_samp=data.first_samp
            )
            return events, fig
        else:
            return events
        
    def split_events(self, events, params, task=None):
        """
        Dispatcher for task-specific event splitting logic.
        """

        if task is None:
            raise ValueError("Task type must be specified")

        if events is None:
            raise ValueError("Events have not been found. Run 'events_finder' first.")

        task_specific_code = {
            "AXCPT": self.split_events_AXCPT,
            "AO": self.split_events_AO
            # add more tasks as needed,
            # "another_task": self._split_events_another_task,
        }

        if task not in task_specific_code:
            raise ValueError(f"Unsupported task type: {task}. Add separate function to object EEGprocessing code.")
        
        return task_specific_code[task](events, params)
    
    def split_events_AXCPT(self, events, params):
        """
        Splits events from the AX-CPT task into different trial types based on cue-target-response sequences.

        Parameters:
        - data: PylineData object containing EEG data and event information.
        - params: Configuration object containing the event_dict and button ID.

        Returns:
        - Tuple of NumPy arrays for each event category:
            AX_cue, AX_target: Trials with A cue and X target, followed by a button press.
            AB_cue, AB_target: Trials with A cue and non-X target, no button press.
            BX_cue, BX_target: Trials with non-A cue and X target, no button press.
            background_AX: B events preceding AX sequences.
            background_A: B events preceding AB sequences.
        """

        # Initialize event category lists
        AX_cue, AX_target = [], []
        AB_cue, AB_target = [], []
        BX_cue, BX_target = [], []
        background_AX, background_A = [], []

        for m in range(len(events) - 2):
            cue, target, next_event = events[m:m+3]
            cue_type, target_type, next_type = cue[2], target[2], next_event[2]

            # A-X successful responses
            if (
                cue_type == params.event_dict['AXCPT']['Filler/4']
                and target_type == params.event_dict['AXCPT']['AX']
                and next_type == params.button_id # only include those AX combinations with a button press
            ):
                AX_cue.append(cue)
                AX_target.append(target)
                
            # A-* successful responses
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/4']
                and target_type == params.event_dict['AXCPT']['Filler/8']
                and next_type != params.button_id # only include those A* combinations with no button press
            ):
                AB_cue.append(cue)
                AB_target.append(target)

            # *-X successful responses
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['*X']
                and next_type != params.button_id # only include those *-X combinations with no button press
            ):
                BX_cue.append(cue)
                BX_target.append(target)
                
            # B-AX (B's preceding AX cue-target presentations)
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['Filler/4']
                and next_type == params.event_dict['AXCPT']['AX'] 
            ):
                background_AX.append(cue)

            # B-AB (B's preceding AB cue-target presentations)
            elif (
                cue_type == params.event_dict['AXCPT']['Filler/8']
                and target_type == params.event_dict['AXCPT']['Filler/4']
                and next_type == params.event_dict['AXCPT']['Filler/8'] 
            ):
                background_A.append(cue)
            

        AX_cue = np.asarray(AX_cue)
        AX_target = np.asarray(AX_target)
        AB_cue = np.asarray(AB_cue)
        AB_target = np.asarray(AB_target)
        BX_cue = np.asarray(BX_cue)
        BX_target = np.asarray(BX_target)
        background_AX = np.asarray(background_AX)
        background_A = np.asarray(background_A)

        return (
              AX_cue, AX_target, AB_cue, AB_target, BX_cue, BX_target, background_AX, background_A
        )
    
    def split_events_AO(self, events, params):
        """
        Categorizes auditory oddball events based on the number of standard tones preceding a target tone.

        Parameters:
        - events: variable containing EEG  events.
        - params: Configuration object with event dictionary.

        Returns:
        - Tuple of arrays: different standard tone conditions (1, 3, 5, etc.) and button responses.
        """

        # Create an array of target tone events which have been responded with a button press
        standard = []
        one_standard= []
        three_standards = []
        five_standards = []
        seven_standards = []
        nine_standards= []
        eleven_standards= []

        responses = [k for k in params.event_dict['AO'].values()]
        responses.remove(32)

        # Iterate through events
        for m in range(len(events) - 1):
            cue, response = events[m:m+2]
            cue_type, response_type = cue[2], response[2]

            # Check for one and three standard tones preceding target tone
            if cue_type == 17 and response_type == 32:
                one_standard.append(cue)
            elif cue_type == 19 and response_type == params.event_dict['AO']['button_id']:
                three_standards.append(cue)
            elif cue_type == 21 and response_type == params.event_dict['AO']['button_id']:
                five_standards.append(cue)
            elif cue_type == 23 and response_type == params.event_dict['AO']['button_id']:
                seven_standards.append(cue)
            elif cue_type == 25 and response_type == params.event_dict['AO']['button_id']:
                nine_standards.append(cue)
            elif cue_type == 27 and response_type == params.event_dict['AO']['button_id']:
                eleven_standards.append(cue)
            elif cue_type == 2: # and response_type in responses:
                standard.append(cue)

        return (
            standard, 
            one_standard,
            three_standards,
            five_standards,
            seven_standards,
            nine_standards, 
            eleven_standards
        )
        
    def create_epochs(self, data, params, tasktype = None, events = None, event_id = None, title = None, epo_duration = None, plot=False):
        """
        Creates EEG epochs either for resting-state (fixed length) or task-based (event-locked).

        Parameters:
        - data (Raw): Raw or filtered EEG data.
        - params: Parameters object with epoch configs.
        - tasktype (str): 'resting' or 'task'.
        - events (array): Event array (only for task-based).
        - event_id (dict): Event dictionary (only for task-based).
        - title (str): Title for the plot (only for task-based).
        - epo_duration (float): Epoch duration in seconds (only for resting).
        - plot (bool): Whether to show epoch image plots.

        Returns:
        - epochs (mne.Epochs or mne.EpochsArray): Created epochs.
        - fig (optional): Epoch plot if `plot=True`.
        """

        if tasktype is None:
                raise ValueError("Must specify either 'resting' or 'task.")
    
        if tasktype == 'resting':
            if epo_duration is None:
                raise ValueError("Duration must be specified for epoching.")
            
            epochs = mne.make_fixed_length_epochs(data, duration=epo_duration, preload=True)

        elif tasktype == 'task':
            if events is None or event_id is None or title is None:
                 raise ValueError("Events, event_id, and title must all be specified for epoching.")
            
            epochs = mne.Epochs(data, events, event_id, tmin=params.epoch_tminmax[0],
                                tmax=params.epoch_tminmax[1], baseline=params.baseline_correction,
                                picks=params.picks, preload=True)
        if plot:
            fig = epochs.plot_image(title=title)
            return epochs, fig
        else:
            return epochs
        
    def reject_auto(self, epochs, method = 'random_search', plot = False):
        """
        Applies automatic rejection to EEG epochs using MNE's rejection threshold and AutoReject.

        Parameters:
        - epochs (mne.Epochs): The epochs to clean.
        - method (str): AutoReject method ('random_search', etc.).
        - plot (bool): Whether to show rejection logs and plots.

        Returns:
        - clean_epochs (mne.Epochs): Cleaned EEG epochs.
        - reject_log: AutoReject's rejection log.
        """
        
        if not isinstance(epochs, (mne.epochs.Epochs, mne.epochs.EpochsArray)):
            raise ValueError ("Reject auto function requires epochs. Run create_epochs.")
        reject_criteria = get_rejection_threshold(epochs)
        print('Dropping epochs with rejection threshold:', reject_criteria)
        
        temp_epochs = epochs.copy()
        temp_epochs.drop_bad(reject=reject_criteria)

        ar = AutoReject(thresh_method=method, random_state=1)
        ar.fit(epochs)
        clean_epochs, reject_log = ar.transform(temp_epochs, return_log=True)
        
        if plot:
            reject_log.plot('horizontal')
            clean_epochs.plot_image(title="GFP with AR ")
            
        return clean_epochs, reject_log
    
    def select_random_epochs(self, epochs, num_epochs_to_select):
        """
        Selects a random group of epochs from an MNE Epochs object.
        
        Parameters:
        - epochs (mne.Epochs): The MNE Epochs object containing all epochs.
        - num_epochs_to_select (int): Number of epochs to randomly select.
        
        Returns:
        - random_epochs (mne.Epochs): The subset of epochs randomly selected.
        """
        # Get the total number of epochs in the Epochs object
        total_epochs = len(epochs)

        if num_epochs_to_select > total_epochs:
            raise ValueError(f'Number of epochs to sub-sample has to be equal to or less than {total_epochs}')
        
        # Generate random indices to select epochs
        random_indices = np.random.choice(total_epochs, size=num_epochs_to_select, replace=False)
        
        # Select the epochs using the random indices
        random_epochs = epochs[random_indices]
    
        return random_epochs