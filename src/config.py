import os

class Config:
    """Central configuration for the Research Engine."""
    
    # API Settings - Default to localhost for single-machine setups
    # Can be overridden via environment variables for WSL2 or remote setups
    API_HOST = os.getenv("ENGINE_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("ENGINE_API_PORT", 8000))
    
    @property
    def api_url(self):
        """Returns the full base URL for the API."""
        return f"http://{self.API_HOST}:{self.API_PORT}"

# Global config instance
config = Config()
