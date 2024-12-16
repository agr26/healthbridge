import yaml
import logging
import os
from pathlib import Path
import argparse
from typing import Dict

# Import pipeline and components
from src.analytics.python.pipeline import HealthcareAnalyticsPipeline
from src.utils.ethics_framework import EthicsFramework
from src.backend.audit.audit_system import SecureAuditSystem

def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    logger = logging.getLogger('HealthBridge')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('healthbridge.log')
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required configuration sections
        required_sections = ['database', 'neural_network', 'ethics', 'audit', 'api']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        return config
        
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def initialize_components(config: Dict, logger: logging.Logger) -> Dict:
    """Initialize all system components."""
    components = {}
    
    try:
        # Initialize Ethics Framework
        logger.info("Initializing Ethics Framework...")
        components['ethics'] = EthicsFramework(config['ethics'])
        
        # Initialize Audit System
        logger.info("Initializing Audit System...")
        components['audit'] = SecureAuditSystem(config['audit'])
        
        # Initialize Analytics Pipeline
        logger.info("Initializing Analytics Pipeline...")
        components['pipeline'] = HealthcareAnalyticsPipeline(config)
        
        return components
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def create_default_config():
    """Create default configuration file if it doesn't exist."""
    default_config = {
        'database': {
            'connection_string': 'postgresql://user:password@localhost:5432/healthcare',
            'pool_size': 5,
            'max_overflow': 10
        },
        'neural_network': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'model_path': 'models/',
            'temporal_steps': 10,
            'temporal_features': 5
        },
        'ethics': {
            'consent_required': True,
            'bias_monitoring': True,
            'audit_enabled': True,
            'privacy_level': 'high'
        },
        'audit': {
            'enabled': True,
            'log_level': 'INFO',
            'retention_days': 90,
            'secure_storage_path': 'audit_logs/'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False,
            'allowed_origins': ['http://localhost:3000'],
            'api_keys_enabled': True
        }
    }
    
    if not os.path.exists('config.yml'):
        with open('config.yml', 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

def verify_system_requirements():
    """Verify system requirements and dependencies."""
    try:
        # Verify required directories exist
        required_dirs = ['models', 'logs', 'data', 'audit_logs']
        for dir_name in required_dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        # Verify database connection
        # Add database connection test here
        
        # Verify model files exist
        # Add model file verification here
        
        return True
        
    except Exception as e:
        raise Exception(f"System requirements verification failed: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='HealthBridge Analytics Platform')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create default config if requested
        if args.create_config:
            create_default_config()
            print("Default configuration file created: config.yml")
            return
        
        # Setup logging
        logger = setup_logging()
        logger.info("Starting HealthBridge Analytics Platform...")
        
        # Load configuration
        config = load_config(args.config)
        if args.debug:
            config['api']['debug'] = True
        
        # Verify system requirements
        logger.info("Verifying system requirements...")
        verify_system_requirements()
        
        # Initialize components
        logger.info("Initializing system components...")
        components = initialize_components(config, logger)
        
        # Start API server
        logger.info("Starting API server...")
        import uvicorn
        uvicorn.run(
            components['pipeline'].app,
            host=config['api']['host'],
            port=config['api']['port'],
            debug=config['api']['debug']
        )
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()