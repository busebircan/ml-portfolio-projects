"""
Deploy Dynamic Pricing Model to Azure ML
Complete deployment script for Azure Machine Learning
"""

import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
    ResourceConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureMLDeployer:
    """Deploy models to Azure ML managed endpoints"""
    
    def __init__(self, 
                 subscription_id: str,
                 resource_group: str,
                 workspace_name: str):
        """
        Initialize Azure ML deployer
        
        Args:
            subscription_id: Azure subscription ID
            resource_group: Resource group name
            workspace_name: Azure ML workspace name
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        # Authenticate
        self.credential = DefaultAzureCredential()
        
        # Initialize ML Client
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info(f"Connected to workspace: {workspace_name}")
    
    def register_model(self,
                      model_path: str,
                      model_name: str,
                      model_version: str = "1.0",
                      description: str = ""):
        """
        Register model in Azure ML
        
        Args:
            model_path: Local path to model directory
            model_name: Name for the registered model
            model_version: Model version
            description: Model description
        
        Returns:
            Registered model object
        """
        logger.info(f"Registering model: {model_name} v{model_version}")
        
        model = Model(
            path=model_path,
            name=model_name,
            version=model_version,
            description=description,
            tags={
                "framework": "scikit-learn",
                "task": "regression",
                "domain": "pricing"
            }
        )
        
        try:
            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"✓ Model registered: {registered_model.name}:{registered_model.version}")
            return registered_model
        
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def create_environment(self,
                          env_name: str,
                          conda_file: str,
                          description: str = ""):
        """
        Create Azure ML environment
        
        Args:
            env_name: Environment name
            conda_file: Path to conda environment file
            description: Environment description
        
        Returns:
            Environment object
        """
        logger.info(f"Creating environment: {env_name}")
        
        env = Environment(
            name=env_name,
            description=description,
            conda_file=conda_file,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            tags={"project": "dynamic-pricing"}
        )
        
        try:
            registered_env = self.ml_client.environments.create_or_update(env)
            logger.info(f"✓ Environment created: {registered_env.name}")
            return registered_env
        
        except Exception as e:
            logger.error(f"Failed to create environment: {str(e)}")
            raise
    
    def create_endpoint(self,
                       endpoint_name: str,
                       description: str = "",
                       auth_mode: str = "key"):
        """
        Create managed online endpoint
        
        Args:
            endpoint_name: Endpoint name (must be unique)
            description: Endpoint description
            auth_mode: Authentication mode ('key' or 'aml_token')
        
        Returns:
            Endpoint object
        """
        logger.info(f"Creating endpoint: {endpoint_name}")
        
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode=auth_mode,
            tags={"project": "dynamic-pricing", "model": "pricing-optimization"}
        )
        
        try:
            created_endpoint = self.ml_client.online_endpoints.begin_create_or_update(
                endpoint
            ).result()
            
            logger.info(f"✓ Endpoint created: {created_endpoint.name}")
            logger.info(f"  Scoring URI: {created_endpoint.scoring_uri}")
            
            return created_endpoint
        
        except ResourceExistsError:
            logger.info(f"Endpoint {endpoint_name} already exists, retrieving...")
            return self.ml_client.online_endpoints.get(endpoint_name)
        
        except Exception as e:
            logger.error(f"Failed to create endpoint: {str(e)}")
            raise
    
    def create_deployment(self,
                         endpoint_name: str,
                         deployment_name: str,
                         model_name: str,
                         model_version: str,
                         environment_name: str,
                         scoring_script: str,
                         code_path: str,
                         instance_type: str = "Standard_DS3_v2",
                         instance_count: int = 1):
        """
        Create deployment under endpoint
        
        Args:
            endpoint_name: Endpoint name
            deployment_name: Deployment name
            model_name: Registered model name
            model_version: Model version
            environment_name: Environment name
            scoring_script: Path to scoring script
            code_path: Path to code directory
            instance_type: VM instance type
            instance_count: Number of instances
        
        Returns:
            Deployment object
        """
        logger.info(f"Creating deployment: {deployment_name}")
        
        # Get registered model
        model = self.ml_client.models.get(name=model_name, version=model_version)
        
        # Get environment
        env = self.ml_client.environments.get(name=environment_name, version="1")
        
        # Configure deployment
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code=code_path,
                scoring_script=scoring_script
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            liveness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                period=10,
                initial_delay=10
            ),
            readiness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                period=10,
                initial_delay=10
            ),
            environment_variables={
                "MODEL_VERSION": model_version,
                "DEPLOYMENT_NAME": deployment_name
            }
        )
        
        try:
            created_deployment = self.ml_client.online_deployments.begin_create_or_update(
                deployment
            ).result()
            
            logger.info(f"✓ Deployment created: {created_deployment.name}")
            
            return created_deployment
        
        except Exception as e:
            logger.error(f"Failed to create deployment: {str(e)}")
            raise
    
    def set_traffic(self, endpoint_name: str, traffic_dict: dict):
        """
        Set traffic distribution across deployments
        
        Args:
            endpoint_name: Endpoint name
            traffic_dict: Traffic distribution (e.g., {"blue": 100})
        """
        logger.info(f"Setting traffic for endpoint: {endpoint_name}")
        
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = traffic_dict
        
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"✓ Traffic set: {traffic_dict}")
    
    def get_endpoint_keys(self, endpoint_name: str):
        """Get endpoint authentication keys"""
        keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
        return keys.primary_key, keys.secondary_key
    
    def test_endpoint(self, endpoint_name: str, sample_data: dict):
        """
        Test endpoint with sample data
        
        Args:
            endpoint_name: Endpoint name
            sample_data: Sample input data
        
        Returns:
            Prediction response
        """
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        import json
        import requests
        
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        primary_key, _ = self.get_endpoint_keys(endpoint_name)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {primary_key}"
        }
        
        response = requests.post(
            endpoint.scoring_uri,
            headers=headers,
            json=sample_data
        )
        
        if response.status_code == 200:
            logger.info("✓ Endpoint test successful")
            return response.json()
        else:
            logger.error(f"Endpoint test failed: {response.text}")
            raise Exception(f"Test failed with status {response.status_code}")


def main():
    """Main deployment workflow"""
    
    # Configuration
    SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
    RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "your-resource-group")
    WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "your-workspace")
    
    ENDPOINT_NAME = "pricing-endpoint"
    DEPLOYMENT_NAME = "blue"
    MODEL_NAME = "dynamic-pricing-model"
    MODEL_VERSION = "1.0"
    ENVIRONMENT_NAME = "pricing-env"
    
    # Paths
    MODEL_PATH = "../../models"
    SCORING_SCRIPT = "score.py"
    CODE_PATH = "."
    CONDA_FILE = "environment.yml"
    
    print("=" * 70)
    print("AZURE ML DEPLOYMENT - DYNAMIC PRICING MODEL")
    print("=" * 70)
    
    try:
        # Initialize deployer
        deployer = AzureMLDeployer(
            subscription_id=SUBSCRIPTION_ID,
            resource_group=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        
        # Step 1: Register model
        print("\n1. Registering model...")
        model = deployer.register_model(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            description="Dynamic pricing optimization model with demand forecasting"
        )
        
        # Step 2: Create environment
        print("\n2. Creating environment...")
        env = deployer.create_environment(
            env_name=ENVIRONMENT_NAME,
            conda_file=CONDA_FILE,
            description="Python environment for dynamic pricing model"
        )
        
        # Step 3: Create endpoint
        print("\n3. Creating endpoint...")
        endpoint = deployer.create_endpoint(
            endpoint_name=ENDPOINT_NAME,
            description="Dynamic pricing prediction endpoint",
            auth_mode="key"
        )
        
        # Step 4: Create deployment
        print("\n4. Creating deployment...")
        deployment = deployer.create_deployment(
            endpoint_name=ENDPOINT_NAME,
            deployment_name=DEPLOYMENT_NAME,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            environment_name=ENVIRONMENT_NAME,
            scoring_script=SCORING_SCRIPT,
            code_path=CODE_PATH,
            instance_type="Standard_DS3_v2",
            instance_count=1
        )
        
        # Step 5: Set traffic
        print("\n5. Setting traffic...")
        deployer.set_traffic(ENDPOINT_NAME, {DEPLOYMENT_NAME: 100})
        
        # Step 6: Get keys
        print("\n6. Retrieving endpoint keys...")
        primary_key, secondary_key = deployer.get_endpoint_keys(ENDPOINT_NAME)
        
        # Step 7: Test endpoint
        print("\n7. Testing endpoint...")
        test_data = {
            "features": [
                {
                    "product_id": "TEST001",
                    "base_price": 100.0,
                    "demand": 50,
                    "inventory": 200,
                    "competitor_price": 105.0,
                    "day_of_week": 1,
                    "month": 6,
                    "is_weekend": 0,
                    "promotional_flag": 0
                }
            ]
        }
        
        result = deployer.test_endpoint(ENDPOINT_NAME, test_data)
        
        # Display results
        print("\n" + "=" * 70)
        print("DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"\nEndpoint Name: {ENDPOINT_NAME}")
        print(f"Scoring URI: {endpoint.scoring_uri}")
        print(f"Primary Key: {primary_key[:20]}...")
        print(f"\nTest Prediction Result:")
        print(f"  Predicted Price: ${result['predictions'][0]:.2f}")
        print(f"  Confidence: {result['confidence'][0]:.2%}")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Save the endpoint URI and primary key securely")
        print("2. Update your application to call the endpoint")
        print("3. Monitor endpoint performance in Azure ML Studio")
        print("4. Set up alerts for model drift and performance")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
