from datetime import datetime
import json
from typing import Dict, Any
from dataclasses import dataclass
import hashlib
import hmac
import logging

@dataclass
class AuditEvent:
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: str
    resource_id: str
    action: str
    outcome: str
    details: Dict[str, Any]
    hash_chain: str

class SecureAuditSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_audit_logging()
        self.last_hash = None

    def _setup_audit_logging(self) -> logging.Logger:
        """Setup secure audit logging with tamper detection."""
        logger = logging.getLogger('SecureAudit')
        logger.setLevel(logging.INFO)
        
        # Create secure file handler
        handler = logging.FileHandler('secure_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(hash)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def log_event(self, event_type: str, user_id: str, 
                 resource_id: str, action: str, 
                 outcome: str, details: Dict) -> None:
        """Log an audit event with tamper-evident chaining."""
        try:
            # Create event
            event = AuditEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                event_type=event_type,
                user_id=user_id,
                resource_id=resource_id,
                action=action,
                outcome=outcome,
                details=details,
                hash_chain=self._calculate_hash_chain(
                    event_type, user_id, resource_id, action
                )
            )
            
            # Log event
            self._secure_log_event(event)
            
            # Update hash chain
            self.last_hash = event.hash_chain
            
        except Exception as e:
            self.logger.error(f"Audit logging error: {str(e)}")
            raise

    def verify_audit_trail(self) -> bool:
        """Verify the integrity of the audit trail."""
        try:
            with open('secure_audit.log', 'r') as f:
                lines = f.readlines()
                
            previous_hash = None
            for line in lines:
                event_hash = line.split(' - ')[1]
                if previous_hash:
                    calculated_hash = self._calculate_hash_chain(
                        previous_hash, 
                        line
                    )
                    if calculated_hash != event_hash:
                        return False
                previous_hash = event_hash
                
            return True
            
        except Exception as e:
            self.logger.error(f"Audit trail verification error: {str(e)}")
            return False

    def _generate_event_id(self) -> str:
        """Generate a unique, tamper-evident event ID."""
        timestamp = datetime.now().isoformat()
        random_bytes = os.urandom(16)
        
        return hashlib.sha256(
            f"{timestamp}{random_bytes}".encode()
        ).hexdigest()

    def _calculate_hash_chain(self, *args) -> str:
        """Calculate a hash chain for tamper evidence."""
        message = '|'.join(str(arg) for arg in args)
        
        if self.last_hash:
            message = f"{self.last_hash}|{message}"
            
        return hmac.new(
            self.config['secret_key'].encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

    def _secure_log_event(self, event: AuditEvent) -> None:
        """Securely log an audit event."""
        event_json = json.dumps({
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'user_id': event.user_id,
            'resource_id': event.resource_id,
            'action': event.action,
            'outcome': event.outcome,
            'details': event.details,
            'hash_chain': event.hash_chain
        })
        
        self.logger.info(
            event_json,
            extra={'hash': event.hash_chain}
        )