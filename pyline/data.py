class PylineData:
    """
    A class to store and manage EEG processing data for the Pyline pipeline.
    
    Attributes:
        raw (Any): Raw EEG data input.
        filtered (Any): Data after filtering step.
        resampled (Any): Data after resampling step.
        ica (Any): Data after Independent Component Analysis (ICA).
        csd (Any): Data after Current Source Density (CSD) transformation.
        pyline (Any): Final processed data for the Pyline pipeline. 
        events (Any): Event-related information associated with the data.
        filename (str): Name of the file associated with the dataset.
        tasktype (str): Type of task associated with the data; either 'task' or 'resting'.
    """

    def __init__(self, raw_data, filename, tasktype = None):
        """
        Initializes the PylineData object with raw data and metadata.

        Args:
            raw_data (Any): The raw EEG dataset to initialize with.
            filename (str): The name of the source file for reference.
            tasktype (str, optional): Type of data ('task' or 'resting'). Default is None.

        Raises:
            ValueError: If tasktype is provided but is not 'task' or 'resting'.
        """

        # Store raw EEG data if provided; otherwise, set to None
        self.raw = raw_data if raw_data else None

        # Placeholders for various stages of EEG data processing
        self.filtered = None
        self.resampled = None
        self.ica = None
        self.csd = None
        self.pyline = None
        self.events = None

        # Store filename if provided; otherwise, set to None
        self.filename = filename if filename else None

        # Validate tasktype if provided, must be either 'task' or 'resting'
        if tasktype is not None and tasktype not in ['task', 'resting']:
            raise ValueError('tasktype must be either "task" or "resting"')

        # Store the task type
        self.tasktype = tasktype
