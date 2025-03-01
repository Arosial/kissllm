from dotenv import load_dotenv

from simplellm.observation import configure_observer

# Load environment variables from .env file if present
load_dotenv()
configure_observer()
