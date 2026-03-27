import os

class EnvironmentManager:
    """Detects the current execution environment and resolves dataset roots dynamically."""
    
    @staticmethod
    def detect_env():
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
            return "kaggle"
        if os.environ.get("GUACAMOLE_ENV") is not None or os.path.exists("/workspaces"):
            return "guacamole"
        return "local"
    
    @staticmethod
    def get_default_data_root(dataset_name="EuroSAT"):
        env = EnvironmentManager.detect_env()
        if env == "kaggle":
            return f"/kaggle/input/{dataset_name.lower()}"
        elif env == "guacamole":
             return f"/data/{dataset_name}"
        else:
             return f"./data/{dataset_name}"
