import pandas as pd
from nir_forensics.preprocessing import snv, savgol, mean_centering

class PreprocessingPipeline():
    """Prepares the data for the Machine Learning Pipeline

        Attributes:
            steps (dict): Dictionary containing the steps of the pipeline, eventually with parameters

        Example:
            >>>

    """
    def __init__(self,
                 steps: dict):
        """Initialize PreprocessingPipeleline with desired steps.

            Args:
                steps (dict): Dictionary containing steps of the Pipeline, if neccessary with parameters

            Example: {'meanCentering' : True, 'snv': True, 'savgol': {'apply' : True, 'derivative' : 1, 'window' : 43}}
        """
        
        self.steps = steps
        self.column_means_ = None

    def fit(self, X: pd.DataFrame):
        """Learn parameters (e.g., column means for mean-centering)

            Args:
                X (pd.DataFrame): Trainng spectra

        """
        if self.steps.get('meanCentering'):
            self.column_means_ = mean_centering(X)
        

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps on X

            Args:
                X (pd.DataFrame): Raw spectra
        """
        X = X.copy()
        snv_step = self.steps.get('snv')
        savgol_params = self.steps.get('savgol', {})
        savgol_step = savgol_params.get('apply', False)
        window = savgol_params.get('window', 11)
        derivative = savgol_params.get('derivative', 1)
        if self.steps.get('meanCentering'):
            X -= self.column_means_
        processed = []
        for i in range(len(X)):
            spectrum = X.iloc[i]
            if snv_step:
                spectrum = snv(spectrum)
            if savgol_step:
                spectrum = savgol(spectrum, derivative = derivative, window = window, apply_snv = False)
            processed.append(spectrum)
        return pd.DataFrame(processed, columns=X.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on X, then transform X."""
        self.fit(X)
        return self.transform(X)

        
    
