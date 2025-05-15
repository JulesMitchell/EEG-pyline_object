class PylineParameters:
    """
    Class for storing and managing EEG analysis parameters.
    These parameters are used across EEG preprocessing, ICA, epoching, event handling, ERP extraction,
    and power spectral density (PSD) analysis.
    """

    def __init__(self, reference=None, eog_channels=None, stimulus_channel=None, filter_design=None, line_noise=None, ica=None,
                 epoch_tminmax=None, baseline_correction=None, event_dict=None, button_id=None, rejection_criteria=None,
                 condition_comp=None, erp_windows=None, channel_picks=None):
        """
        Initialize an instance of PylineParameters with defaults or user-defined values.

        Parameters:
        - reference (str): EEG referencing scheme, e.g., 'average', 'mastoids', 'Cz'.
        - eog_channels (list of str): Names of channels used for EOG artifact rejection.
        - stimulus_channel (str): Channel name that carries stimulus trigger codes.
        - filter_design (dict): Dictionary of band-pass filter settings.
        - line_noise (float or None): Line noise frequency to be removed (e.g., 60Hz).
        - ica (dict): Parameters for ICA decomposition.
        - epoch_tminmax (list of float): Time window around events to epoch the data.
        - baseline_correction (tuple or None): Time range for baseline correction.
        - event_dict (dict): Dictionary mapping condition names to event IDs.
        - button_id (int): Event code corresponding to a participant response/button press.
        - rejection_criteria (dict): Criteria for rejecting epochs, e.g., voltage thresholds.
        - condition_comp (list): Conditions to be compared for ERP analysis.
        - erp_windows (dict): ERP component windows with their polarity.
        - channel_picks (list of str): Channels used for ERP feature extraction.
        """
        self.reference = reference if reference is not None else 'average'
        self.projector = True if self.reference == 'average' else False

        # Default EOG channel names typically used in Biosemi setups
        self.eog_channels = eog_channels if eog_channels is not None else [
            'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'
        ]

        # Stimulus channel name (Biosemi usually uses 'Status')
        self.stimulus_channel = stimulus_channel if stimulus_channel is not None else 'Status'

        # Default bandpass filter design for EEG preprocessing
        self.filter_design = filter_design if filter_design is not None else {
            'l_freq': 0.4, 'h_freq': 30, 'filter_length': 'auto',
            'l_trans_bandwidth': 'auto', 'h_trans_bandwidth': 'auto', 'n_jobs': None,
            'method': 'fir', 'iir_params': None, 'phase': 'zero',
            'fir_window': 'hamming', 'fir_design': 'firwin', 'pad': 'reflect_limited'
        }

        # Line noise frequency to remove (e.g., 60Hz) – optional
        self.line_noise = line_noise if line_noise else None

        # ICA (Independent Component Analysis) configuration
        self.ica = ica if ica is not None else {
            'use': False,
            'n_components': 15,
            'method': 'fastica',
            'plot': False,
            'autoselect': False
        }

        # Epoch window in seconds, relative to event onset
        self.epoch_tminmax = epoch_tminmax if epoch_tminmax is not None else [-0.2, 0.8]

        # Time window for baseline correction (can be None)
        self.baseline_correction = baseline_correction if baseline_correction is not None else None

        # Dictionary of event codes used for different tasks
        self.event_dict = event_dict if event_dict is not None else {
            'AXCPT': {'Filler/4': 4, 'Filler/8': 8, 'AX': 2, '*X': 6, 'button_id': 128},
            'AO': {
                'target after 1 standard': 17, 'target after 3 standards': 19,
                'target after 5 standards': 21, 'target after 7 standards': 23,
                'target after 9 standards': 25, 'target after 11 standards': 27, 'button_id': 32
            },
            'resting': {'Initiated': 1, 'Start Tone': 4, 'End Tone': 2}
        }

        # Default button press event ID (e.g., 128 for Biosemi trigger)
        self.button_id = button_id if button_id is not None else 128

        # Epoch rejection thresholds (in volts); 150 µV = 0.00015 V
        self.rejection_criteria = rejection_criteria if rejection_criteria is not None else {'eeg': 0.00015}

        # Conditions used for ERP comparison or statistical tests
        self.condition_comp = condition_comp if condition_comp is not None else ["AX", "*X"]

        # ERP component windows (in ms) and expected polarity: 1 = positive, -1 = negative
        self.erp_windows = erp_windows if erp_windows is not None else {
            'N1': [40, 170, -1],
            'N2': [180, 350, -1],
            'P2': [100, 260, 1],
            'P3': [270, 500, 1]
        }

        # EEG channels of interest for ERP extraction
        self.channel_picks = channel_picks if channel_picks is not None else ['Fz', 'Cz', 'Pz']

        # Optionally defined later; used for channel selection
        self.picks = None

        # Configuration for PSD (Power Spectral Density) analysis
        self.psd_options = {
            'bands': {
                'Delta': [1, 3.9],
                'Theta': [4, 7.9],
                'Alpha': [8, 12],
                'Beta': [12.1, 30]
            },
            'brain_regions': {
                'Left frontal': ['AF3', 'F3', 'FC1'],
                'Right frontal': ['AF4', 'F4', 'FC2'],
                'Left temporal': ['F7', 'FC5', 'T7'],
                'Right temporal': ['F8', 'FC6', 'T8'],
                'Left posterior': ['CP5', 'P3', 'P7'],
                'Right posterior': ['CP6', 'P4', 'P8']
            },
            'params': {
                'method': 'welch',
                'fminmax': [1, 30],
                'window': 'hamming',
                'window_duration': 2.5,
                'window_overlap': 0.5,
                'zero_padding': 3
            }
        }