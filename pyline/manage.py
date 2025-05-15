# import packages
import mne, os, warnings
import pandas as pd
from pathlib import Path
import pickle

## Issue log: .fif file extensions can be used for various filetypes (e.g. evoked or raw). This creates an issue when the file_type_to_funciton is called. 
class PylineManage:
    """
    Class for managing EEG data directory structure, loading data, and processing information
    using the MNE-Python framework. Supports BIDS-like data structures and classification utilities.
    """

    def __init__(self, root_folder=None, exp_folder=None, clean_folder=None, eeg_params=None):
        """
        Initialize PylineManage with paths and EEG loading options.

        Parameters:
        - root_folder (str): Base directory containing subject/session folders.
        - exp_folder (str): Subfolder for a specific subject or experiment (default 'sub-01').
        - clean_folder (str): Directory to export or store cleaned EEG data (default 'derivatives/').
        - eeg_params (object): Parameters used for EEG preprocessing (e.g., references, EOG, stim channels).
        """
        self.root_folder = root_folder if root_folder is not None else 'test/'
        self.exp_folder = exp_folder if exp_folder is not None else 'sub-01'
        self.clean_folder = clean_folder if clean_folder is not None else 'derivatives/'

        # Mapping of EEG file types to MNE functions
        self.file_type_to_function = {
            'bdf': mne.io.read_raw_bdf,
            'fif': mne.io.read_raw_fif,
            'edf': mne.io.read_raw_edf,
            'set': mne.io.read_raw_eeglab,
            'vhdr': mne.io.read_raw_brainvision,  # EOG channels need to be specified explicitly if required
        }

        # Supported montage (channel layout) options
        self.montage_options = {
            'biosemi32': mne.channels.make_standard_montage('biosemi32'),
            'standard_1020': mne.channels.make_standard_montage('standard_1020')
        }

        # EEG reference schemes
        self.reference_options = {
            'average': 'average',
            'mastoids': ['EXG1', 'EXG2'],
            'Cz': ['Cz']
        }

    def array_to_df(self, subject, epochs, array_channels):
        """
        Convert channel data array into a DataFrame with labeled columns and subject index.

        Parameters:
        - subject (str): Name or ID of the subject.
        - epochs (mne.Epochs): MNE Epochs object for channel metadata.
        - array_channels (array-like): Channel-wise data (e.g., PSD values).

        Returns:
        - df_channels (pd.DataFrame): DataFrame of channel data with subject index.
        """
        df_channels = pd.DataFrame(array_channels).T
        try:
            df_channels.columns = epochs.info.ch_names[:32]  # Assign channel names (limit to 32 if needed)
        except Exception as e:
            print(f"The number of channels for {subject} is > expected from psd calculations: {e}")
        df_channels['Subject'] = subject
        df_channels.set_index('Subject', inplace=True)
        return df_channels

    def df_channels_to_regions(self, df_psd_band, brain_regions):
        """
        Aggregate channel-level data into predefined brain regions.

        Parameters:
        - df_psd_band (pd.DataFrame): PSD values per channel and subject.
        - brain_regions (dict): Mapping of brain region names to associated channel names.

        Returns:
        - df_psd_reg_band (pd.DataFrame): Region-averaged PSD values per subject.
        """
        df_psd_reg_band = pd.DataFrame()
        for region in brain_regions:
            df_temp = df_psd_band[brain_regions[region]].copy().mean(axis=1)
            df_psd_reg_band = pd.concat([df_psd_reg_band, df_temp], axis=1)
        df_psd_reg_band.columns = brain_regions.keys()
        df_psd_reg_band.index.name = 'Subject'
        return df_psd_reg_band

    def get_dirs(self, extension='.bdf'):
        """
        Retrieve full paths and subject names for EEG files of a specific extension.

        Parameters:
        - extension (str): File extension to filter EEG files (default '.bdf').

        Returns:
        - file_dirs (list of str): Full file paths of EEG data.
        - subject_names (list of str): Names of subjects based on filenames.
        """
        dir_inprogress = os.path.join(self.root_folder, self.exp_folder)
        file_dirs = [os.path.join(dir_inprogress, f) for f in os.listdir(dir_inprogress) if f.endswith(extension)]
        subject_names = [os.path.splitext(f)[0] for f in os.listdir(dir_inprogress) if f.endswith(extension)]
        return file_dirs, subject_names

    def extract_string_from_path(self, file_path, string=None):
        """
        Extract a matching substring from a file path.

        Parameters:
        - file_path (str): Full file path.
        - string (str): Substring to search for in the path.

        Returns:
        - str or None: The matching string if found, else None.

        Raises:
        - ValueError: If `string` is not provided.
        """
        if string is None:
            raise ValueError("Must provide string to extract.")
        file_path_lower = file_path.lower()
        if string and string.lower() in file_path_lower:
            return string
        else:
            return None

    def extract_file_info(self, path, extension, string=None):
        """
        Collect metadata and file paths from a session folder.

        Parameters:
        - path (str): Directory containing EEG files.
        - extension (str): File extension of EEG files.
        - string (str, optional): Task identifier string to extract from filenames.

        Returns:
        - bids_files (list of dict): Metadata and file paths for each EEG file.
        """
        bids_files = []
        try:
            files = [f for f in os.listdir(path) if f.endswith(extension)]
        except OSError as e:
            raise RuntimeError(f"Error accessing eeg folder: {path} - {e}")
        for file in files:
            file_path = os.path.join(path, file)
            task = self.extract_string_from_path(file_path, string=string) if string else 'Not Specified'
            bids_files.append({
                "subject": os.path.basename(os.path.dirname(os.path.dirname(path))),
                "session": os.path.basename(os.path.dirname(path)),
                "modality": os.path.basename(path),
                "task": task,
                "data": file_path
            })
        return bids_files

    def get_bids(self, modalities, extensions, str_to_extract=None, include=None, exclude=None):
        """
        Traverse BIDS-style EEG folder structure and compile dataset metadata.

        Parameters:
        - modalities (list of str): EEG subfolder names per session.
        - extensions (list of str): File extensions for each modality.
        - str_to_extract (list of str): Optional substrings to detect task type in file paths.
        - include (list of str): Specific subjects to include.
        - exclude (list of str): Specific subjects to exclude.

        Returns:
        - bids_files (list of dict): Extracted metadata and paths for EEG files.
        """
        if not isinstance(modalities, list):
            raise TypeError("The `modalities` argument must be a list.")
        if not isinstance(extensions, list):
            raise TypeError("The `extensions` argument must be a list.")
        if str_to_extract is not None and not isinstance(str_to_extract, list):
            raise TypeError("The `str_list` argument must be a list.")
        if str_to_extract is not None and len(str_to_extract) != len(modalities):
            raise ValueError("The `str_list` argument must be the same length as the `modalities` argument.")
        if len(extensions) != len(modalities):
            raise ValueError("The `extensions` argument must be the same length as the `modalities` argument.")
        if include and exclude:
            raise ValueError("Cannot specify both 'include' and 'exclude' at the same time.")

        include = include if include else []
        exclude = exclude if exclude else []

        bids_files = []
        try:
            subject_folders = os.listdir(self.root_folder)
        except OSError as e:
            print(f"Error accessing raw folder: {self.root_folder} - {e}")
            return bids_files

        for subject_folder in subject_folders:
            if subject_folder.lower() in ["derivatives", "analysis"]:
                continue
            if include and subject_folder not in include:
                continue
            if exclude and subject_folder in exclude:
                continue

            subject_path = os.path.join(self.root_folder, subject_folder)
            try:
                session_folders = os.listdir(subject_path)
            except OSError as e:
                print(f"Error accessing subject folder: {subject_path} - {e}")
                continue

            for session_folder in session_folders:
                session_path = os.path.join(subject_path, session_folder)
                for i, modality_folder in enumerate(modalities):
                    modality_path = os.path.join(session_path, modality_folder)
                    bids_files.extend(self.extract_file_info(
                        modality_path,
                        extension=extensions[i],
                        string=str_to_extract[i] if str_to_extract else None
                    ))

        return bids_files

    def load_eeg(self, file, params, montage='biosemi32'):
        """
        Load and preprocess EEG data using MNE.

        Parameters:
        - file (str): Path to EEG data file.
        - params (object): Object containing EEG loading parameters (eog_channels, stimulus_channel, reference, projector).
        - montage (str): Channel layout name (default 'biosemi32').

        Returns:
        - raw (mne.io.Raw): Loaded and referenced raw EEG data.
        """
        file_extension = file.split('.')[-1]

        if file_extension not in self.file_type_to_function:
            raise ValueError(f"Unsupported file type: {file_extension}")

        try:
            raw = self.file_type_to_function[file_extension](file, infer_types=True, eog=params.eog_channels,
                                                             stim_channel=params.stimulus_channel)
        except Exception:
            try:
                raw = self.file_type_to_function[file_extension](file, eog=params.eog_channels)
            except Exception:
                try:
                    raw = self.file_type_to_function[file_extension](file)
                except Exception as e:
                    raise ValueError(f"Error loading EEG data: {e}. Check file_type_to_function configuration.")

        if montage not in self.montage_options:
            raise ValueError(f"Unsupported montage: {montage}")
        if params.reference not in self.reference_options:
            raise ValueError(f"Unsupported reference: {params.reference}")

        if file_extension != 'fif':
            try:
                raw = raw.set_montage(self.montage_options[montage], on_missing='ignore').load_data()
                raw, _ = mne.set_eeg_reference(
                    raw, ref_channels=self.reference_options[params.reference],
                    projection=params.projector, verbose=True, copy=True)
            except Exception as e:
                print(f"Warning: Error setting montage or reference: {e}. Proceeding without those steps.")

        return raw

    def create_classifer_file(self, file_path, classifier):
        """
        Save a classifier mapping dictionary to a file.

        Parameters:
        - file_path (str): Destination path for the file.
        - classifier (dict): Dictionary of classifier assignments.
        """
        with open(file_path, 'w') as file:
            for key, value in classifier.items():
                file.write(f'{key}: {value}\n')

    def load_classifier_file(self, file_path):
        """
        Load a classifier mapping dictionary from a file.

        Parameters:
        - file_path (str): Path to the classifier file.

        Returns:
        - loaded_mapping (dict): Dictionary loaded from file.
        """
        loaded_mapping = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.split(':', 1)
                loaded_mapping[key.strip()] = eval(value.strip())
        return loaded_mapping

    def add_classifier(self, bids, classifier, designation=[1, 0]):
        """
        Assign group labels to BIDS entries based on classification mapping.

        Parameters:
        - bids (list of dict): List of BIDS entries.
        - classifier (dict): Nested dict with session-based subject classifications.
        - designation (list): Two-item list, e.g., [1 (grouped), 0 (not grouped)].
        """
        for entry in bids:
            participant = entry['subject']
            timepoint = entry['session']
            found_designation = designation[1]
            if timepoint in classifier:
                if participant in classifier[timepoint]:
                    found_designation = designation[0]
            entry['group'] = found_designation

    def load_electrodes_file(self, file_path):
        """
        Load a list of electrode/channel names from a text file.

        Parameters:
        - file_path (str): Path to the electrode list file.

        Returns:
        - electrodes (list of str): List of channel names.
        """
        electrodes = []
        with open(file_path, 'r') as file:
            for line in file:
                electrodes.append(line.strip())
        return electrodes

## The following is old code - can delete once changes confirmed
# class PylineManage:
#     def __init__(self, root_folder=None, exp_folder=None, clean_folder=None, eeg_params = None):
#         """
#         Initialize EEG Directories.

#         Parameters:
#         - raw_folder (str): Directory containing raw EEG files.
#         - exp_folder (str): Experiment folder within the raw folder.
#         - clean_folder (str): Directory to export clean files.
#         """
#         self.root_folder = root_folder if root_folder is not None else 'test/'
#         self.exp_folder = exp_folder if exp_folder is not None else 'sub-01'
#         self.clean_folder = clean_folder if clean_folder is not None else 'derivatives/'
#         self.file_type_to_function = {
#             'bdf': mne.io.read_raw_bdf,
#             'fif': mne.io.read_raw_fif,
#             'edf': mne.io.read_raw_edf,
#             'set': mne.io.read_raw_eeglab,
#             'vhdr': mne.io.read_raw_brainvision, # Need to determine how EOG channels need to be specified. Defaults are: eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto'
#             # 'fif': mne.read_evokeds # change to evoke
#         }
#         self.montage_options = {
#             'biosemi32': mne.channels.make_standard_montage('biosemi32'),
#             'standard_1020': mne.channels.make_standard_montage('standard_1020')
#         }
#         self.reference_options = {
#             'average': 'average',
#             'mastoids': ['EXG1', 'EXG2'],
#             'Cz': ['Cz']
#         }

#     def array_to_df(self, subject, epochs, array_channels):
#         """
#         Convert channel-based array to Pandas dataframe with channels' and subjects' names.

#         Parameters
#         ----------
#         subjectname: A string for subject's name
#         epochs: Epochs-type (MNE-Python) EEG file
#         array_channels: An array with values for each channel

#         Returns
#         -------
#         df_channels: A dataframe with values for each channel
#         """
#         df_channels = pd.DataFrame(array_channels).T
#         try:
#             df_channels.columns = epochs.info.ch_names[:32]
#         except Exception as e:
#             print(f"The number of channels for {subject} is > expected from psd calculations: {e}")
#         df_channels['Subject'] = subject
#         df_channels.set_index('Subject', inplace=True)

#         return df_channels

#     def df_channels_to_regions(self, df_psd_band, brain_regions):
#         """
#         Average channels together based on the defined brain regions.

#         Parameters
#         ----------
#         df_psd_band: A dataframe with PSD values for each channel per subject
#         brain_regions: A dictionary of brain regions and EEG channels which they contain
#         drop_cols: List of columns which are not channel PSD data

#         Returns
#         -------
#         df_psd_reg_band: A dataframe with PSD values for each brain region per subject
#         """

#         df_psd_reg_band = pd.DataFrame()
#         for region in brain_regions:
#             df_temp = df_psd_band[brain_regions[region]].copy().mean(axis=1)
#             df_psd_reg_band = pd.concat([df_psd_reg_band, df_temp], axis=1)
            
#         df_psd_reg_band.columns = brain_regions.keys()
#         df_psd_reg_band.index.name = 'Subject'

#         return df_psd_reg_band
        
#     def get_dirs(self, extension='.bdf'):
#         """
#         Get directories of raw EEG files by specifying appropriate extension.

#         Parameters:
#         - extension (str): File extension to filter files. Default 'bdf'.

#         Returns:
#         - file_dirs (list of str): List of file paths to the EEG data files.
#         - subject_names (list of str): List of subject names.
#         """
#         dir_inprogress = os.path.join(self.root_folder, self.exp_folder)
#         file_dirs = [os.path.join(dir_inprogress, f) for f in os.listdir(dir_inprogress) if f.endswith(extension)]
#         subject_names = [os.path.splitext(f)[0] for f in os.listdir(dir_inprogress) if f.endswith(extension)]
#         return file_dirs, subject_names

#     def extract_string_from_path(self, file_path, string=None):
#         """
#         Extract the first matching string from the file path.
#         This can be modified to handle different naming conventions.

#         Args:
#             file_path (str): The file path from which to extract the string.
#             str1 (str, optional): The first string to search for in the file path.
#             str2 (str, optional): The second string to search for in the file path.

#         Returns:
#             str: The first matching string found in the file path, or None if no match is found.

#         Raises:
#             ValueError: If neither `str1` nor `str2` is provided.
#             ValueError: If both `str1` and `str2` are empty strings.
#         """
#         if string is None :
#             raise ValueError("Must provide string to extract.")

#         file_path_lower = file_path.lower()

#         if string and string.lower() in file_path_lower:
#             return string
#         else:
#             return None

#     def extract_file_info(self, path, extension, string=None):
#         """
#         Process a session folder and extract the relevant information.
#         """
#         bids_files = []

#         try:
#             files = [f for f in os.listdir(path) if f.endswith(extension)]
#         except OSError as e:
#             raise RuntimeError(f"Error accessing eeg folder: {path} - {e}")

#         for file in files:
#             file_path = os.path.join(path, file)
#             if string:
#                 task = self.extract_string_from_path(file_path, string=string)
#                 bids_files.append({
#                     "subject": os.path.basename(os.path.dirname(os.path.dirname(path))),
#                     "session": os.path.basename(os.path.dirname(path)),
#                     "modality": os.path.basename(path),
#                     "task": task,
#                     "data": file_path
#                 })
#             else:
#                 bids_files.append({
#                     "subject": os.path.basename(os.path.dirname(os.path.dirname(path))),
#                     "session": os.path.basename(os.path.dirname(path)),
#                     "modality": os.path.basename(path),
#                     "task": 'Not Specified',
#                     "data": file_path
#                 })

#         return bids_files
    
#     def get_bids(self, modalities, extensions, str_to_extract=None, include=None, exclude=None):
#         """
#         Using the (BIDS) file structure, load relevant information (subject, session, task-type, file location) for raw eeg files.
#         The `extensions` and `str_list` arguments are lists that must be the same length as `modalities`.
#         Finally, specify specific participants to include or exclude from the analysis.
#         """

#         if not isinstance(modalities, list):
#             raise TypeError("The `modalities` argument must be a list.")

#         if not isinstance(extensions, list):
#             raise TypeError("The `extensions` argument must be a list.")

#         if str_to_extract is not None and not isinstance(str_to_extract, list):
#             raise TypeError("The `str_list` argument must be a list.")

#         if str_to_extract is not None and len(str_to_extract) != len(modalities):
#             raise ValueError("The `str_list` argument must be the same length as the `modalities` argument.")

#         if len(extensions) != len(modalities):
#             raise ValueError("The `extensions` argument must be the same length as the `modalities` argument.")

#         if include and exclude:
#             raise ValueError("Cannot specify both 'include' and 'exclude' at the same time.")

#         include = include if include else []
#         exclude = exclude if exclude else []

#         bids_files = []

#         try:
#             subject_folders = os.listdir(self.root_folder)
#         except OSError as e:
#             print(f"Error accessing raw folder: {self.root_folder} - {e}")
#             return bids_files

#         for subject_folder in subject_folders:
#             # Skip the subject folder if it's "derivatives" or "analysis", or if user specified
#             if subject_folder.lower() in ["derivatives", "analysis"]:
#                 continue
#             if include and subject_folder not in include:
#                 continue
#             if exclude and subject_folder in exclude:
#                 continue

#             subject_path = os.path.join(self.root_folder, subject_folder)

#             try:
#                 session_folders = os.listdir(subject_path)
#             except OSError as e:
#                 print(f"Error accessing subject folder: {subject_path} - {e}")
#                 continue

#             for session_folder in session_folders:
#                 session_path = os.path.join(subject_path, session_folder)

#                 for i, modality_folder in enumerate(modalities):
#                     modality_path = os.path.join(session_path, modality_folder)
#                     bids_files.extend(self.extract_file_info(modality_path, extension=extensions[i], string=str_to_extract[i] if str_to_extract else None))

#         return bids_files
    
#     def load_eeg(self, file, params, montage='biosemi32'):
#         """
#         Load EEG data from file directory using mne import function based on file extension.

#         Parameters:
#         - file (str): Path to the EEG data file.
#         - montage (str): Data montage (channel set-up) to apply to the data file. Default 'biosemi32'.
#         - reference (str): Reference to apply to the data file. Default 'average'.

#         Returns:
#         - raw_data (mne.io.Raw): Loaded raw EEG data object.
#         """
#         file_extension = file.split('.')[-1]

#         if file_extension not in self.file_type_to_function:
#             raise ValueError(f"Unsupported file type: {file_extension}")
        
#         try:
#             raw = self.file_type_to_function[file_extension](file, infer_types=True, eog=params.eog_channels,
#                                                             stim_channel=params.stimulus_channel)
#         except Exception as e:
#             try:
#                 raw = self.file_type_to_function[file_extension](file, eog=params.eog_channels)
#             except Exception as e:
#                 try:
#                     raw = self.file_type_to_function[file_extension](file)
#                 except Exception as e:
#                     raise ValueError(f"Error loading EEG data: {e}. File type {file_extension} does not have arguments specified correctly. Review 'file_type_to_function' attribute of manage object.")

#         if montage not in self.montage_options:
#             raise ValueError(f"Unsupported montage: {montage}")
#         if params.reference not in self.reference_options:
#             raise ValueError(f"Unsupported reference: {params.reference}")

#         if file_extension != 'fif':
#             try:
#                 raw = raw.set_montage(self.montage_options[montage], on_missing = 'ignore').load_data()

#                 raw, ref_data = mne.set_eeg_reference(raw, ref_channels=self.reference_options[params.reference], projection = params.projector, verbose=True, copy = True)
                
#             except Exception as e:
#                 print(f"Warning: Error setting montage or reference: {e}. Proceeding without applying montage and reference.")
    
#         return raw
    
#     def create_classifer_file(self, file_path, classifier):
#         # Open the file in write mode ('w')
#         with open(file_path, 'w') as file:
#             # Write the dictionary to the file
#             for key, value in classifier.items():
#                 file.write(f'{key}: {value}\n')

#     def load_classifier_file(self, file_path):
#         # Initialize an empty dictionary to store the loaded data
#         loaded_mapping = {}

#         # Open the file in read mode ('r')
#         with open(file_path, 'r') as file:
#             # Read each line in the file
#             for line in file:
#                 # Split each line by the first occurrence of ':' to separate key and value
#                 key, value = line.split(':', 1)
#                 # Remove any leading/trailing whitespaces and convert value to list
#                 loaded_mapping[key.strip()] = eval(value.strip())

#         return loaded_mapping
    
#     def add_classifier(self, bids, classifier, designation=[1, 0]):

#         for entry in bids:
#             participant = entry['subject']
#             timepoint = entry['session']
            
#             found_designation = designation[1]  # Default designation if not found
            
#             # Check if the participant is in the response_status_mapping for the given timepoint
#             if timepoint in classifier:
#                 if participant in classifier[timepoint]:
#                     found_designation = designation[0]
            
#             # Assign the found designation to the bid
#             entry['group'] = found_designation

#     def load_electrodes_file(self, file_path):
#     # Initialize an empty dictionary to store the loaded data
#         electrodes = []

#         # Open the file in read mode ('r')
#         with open(file_path, 'r') as file:
#             # Read each line in the file
#             for line in file:
#                 # Strip any extra whitespace (like newline characters) and add to the list
#                 electrodes.append(line.strip())

#         return electrodes
    
    # def idtxl_combinetargets(folder, threshold=2, contains='placeholder', output_path='derivatives'):
    #     """
    #     Combines single tartet IDTxL network results from pickel files in specified folders and saves the results as combined network pickle files.

    #     Parameters:
    #     - folder (str or Path): Path to the root folder containing subject and session folders (BIDs format).
    #     - threshold (int): Threshold value used in network_fdr function. Threshold 1 is more lenient than 2. See IDTxL documentation.
    #     - contains (str): Substring to filter files by.
    #     - output_path (str or Path): Path where the output pickle files will be saved.

    #     Returns:
    #     - res: Result from network_fdr function.
    #     """

    #     # Convert folder and output_path to Path objects if they are strings
    #     folder = Path(folder)
    #     output_path = Path(output_path)

    #     # Initialize res as None to handle case where no files are processed
    #     res = None

    #     # Loop through subject folders
    #     for subject_folder in folder.iterdir():
    #         if not subject_folder.is_dir():
    #             continue
            
    #         if subject_folder.name == 'derivatives':  # Compare against 'name' attribute
    #             continue

    #         # Loop through session folders
    #         for session_folder in subject_folder.iterdir():
    #             if not session_folder.is_dir():
    #                 continue

    #             # Filter files based on whether the file name contains the 'contains' substring
    #             files = [f for f in session_folder.iterdir() if f.is_file() and contains in f.name]

    #             # Skip processing if there are no files to combine
    #             if not files:
    #                 print(f'No files containing "{contains}" found in {session_folder}')
    #                 continue

    #             # Load results using pickle
    #             res_list = []

    #             for file in files:
    #                 with open(file, 'rb') as f:
    #                     res_list.append(pickle.load(f))

    #             # Perform operation to combine targets
    #             res = network_fdr({'alpha_fdr': 0.05, 'fdr_constant': threshold}, *res_list)

    #             # Construct the output file path using pathlib
    #             output_dir = folder / output_path / subject_folder.name / session_folder.name
    #             output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    #             output_file = f'{subject_folder.name}_{session_folder.name}_{contains}_network.p'
    #             file_path = output_dir / output_file
                
    #             # Save combined results as pickle file
    #             with open(file_path, 'wb') as f:
    #                 pickle.dump(res, f)

    #             print(f'Successfully Combined Targets for {subject_folder.name}/{session_folder.name}')
    #     return 