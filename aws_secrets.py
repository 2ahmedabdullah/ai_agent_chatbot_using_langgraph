import json
import logging
import boto3
from botocore.exceptions import ClientError
import os
 
logger = logging.getLogger(__name__)
 
def load_aws_secrets():
    """
    Fetch secrets from AWS Secrets Manager and inject into os.environ.
    All existing os.getenv() calls work without any changes.
    """
    secret_name = "prod/company/openai-chatbot"
    region_name = "ap-southeast-1"
 
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response['SecretString'])
         # DEBUG: Print all loaded keys
        logger.info("AWS Secrets loaded (%d keys)", len(secrets))
        logger.info("Keys found: %s", list(secrets.keys()))  # ← Add this line
 
        # Inject into environment
        for key, value in secrets.items():
            os.environ[key] = str(value)
 
        logger.info("AWS Secrets loaded (%d keys)", len(secrets))
 
    except ClientError as e:
        logger.error("Failed to load AWS Secrets: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading secrets: %s", e)
        raise