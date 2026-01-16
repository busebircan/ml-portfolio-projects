"""
Azure Database Configuration and Connector
Handles connections to Azure SQL Database, Cosmos DB, and Blob Storage
"""

import os
import pyodbc
import pandas as pd
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureDataConnector:
    """
    Unified connector for Azure data services
    Supports: Azure SQL Database, Cosmos DB, Blob Storage
    """
    
    def __init__(self, use_managed_identity: bool = True):
        """
        Initialize Azure connector with authentication
        
        Args:
            use_managed_identity: Use Azure Managed Identity (True for production)
        """
        self.use_managed_identity = use_managed_identity
        
        if use_managed_identity:
            self.credential = ManagedIdentityCredential()
        else:
            self.credential = DefaultAzureCredential()
        
        # Initialize Key Vault client if vault URL is provided
        self.kv_client = None
        if os.getenv('KEY_VAULT_URL'):
            self.kv_client = SecretClient(
                vault_url=os.getenv('KEY_VAULT_URL'),
                credential=self.credential
            )
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault"""
        if self.kv_client:
            return self.kv_client.get_secret(secret_name).value
        return os.getenv(secret_name)
    
    def get_sql_connection(self, 
                          server: Optional[str] = None,
                          database: Optional[str] = None,
                          use_aad_auth: bool = True):
        """
        Connect to Azure SQL Database
        
        Args:
            server: Azure SQL server name (e.g., 'myserver.database.windows.net')
            database: Database name
            use_aad_auth: Use Azure AD authentication (recommended)
        
        Returns:
            pyodbc connection object
        """
        server = server or os.getenv('AZURE_SQL_SERVER')
        database = database or os.getenv('AZURE_SQL_DATABASE')
        
        if not server or not database:
            raise ValueError("SQL server and database must be provided")
        
        if use_aad_auth:
            # Azure AD authentication (recommended for production)
            connection_string = f"""
                Driver={{ODBC Driver 17 for SQL Server}};
                Server=tcp:{server},1433;
                Database={database};
                Authentication=ActiveDirectoryMsi;
                Encrypt=yes;
                TrustServerCertificate=no;
                Connection Timeout=30;
            """
        else:
            # SQL authentication (for development)
            username = self.get_secret('SQL_USERNAME')
            password = self.get_secret('SQL_PASSWORD')
            connection_string = f"""
                Driver={{ODBC Driver 17 for SQL Server}};
                Server=tcp:{server},1433;
                Database={database};
                Uid={username};
                Pwd={password};
                Encrypt=yes;
                TrustServerCertificate=no;
                Connection Timeout=30;
            """
        
        try:
            conn = pyodbc.connect(connection_string)
            logger.info(f"Successfully connected to Azure SQL: {database}")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to Azure SQL: {str(e)}")
            raise
    
    def load_data_from_sql(self, 
                          query: str,
                          params: Optional[tuple] = None,
                          server: Optional[str] = None,
                          database: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from Azure SQL Database
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            server: Azure SQL server name
            database: Database name
        
        Returns:
            pandas DataFrame with query results
        """
        conn = self.get_sql_connection(server, database)
        
        try:
            if params:
                df = pd.read_sql(query, conn, params=params)
            else:
                df = pd.read_sql(query, conn)
            
            logger.info(f"Loaded {len(df)} rows from Azure SQL")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data from SQL: {str(e)}")
            raise
        
        finally:
            conn.close()
    
    def write_predictions_to_sql(self,
                                df: pd.DataFrame,
                                table_name: str,
                                server: Optional[str] = None,
                                database: Optional[str] = None,
                                if_exists: str = 'append'):
        """
        Write predictions to Azure SQL Database
        
        Args:
            df: DataFrame with predictions
            table_name: Target table name
            server: Azure SQL server name
            database: Database name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        conn = self.get_sql_connection(server, database)
        
        try:
            # Use pandas to_sql for efficient bulk insert
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Successfully wrote {len(df)} predictions to {table_name}")
        
        except Exception as e:
            logger.error(f"Failed to write predictions to SQL: {str(e)}")
            raise
        
        finally:
            conn.close()
    
    def execute_sql(self,
                   query: str,
                   params: Optional[tuple] = None,
                   server: Optional[str] = None,
                   database: Optional[str] = None):
        """
        Execute SQL statement (INSERT, UPDATE, DELETE, etc.)
        
        Args:
            query: SQL statement
            params: Query parameters
            server: Azure SQL server name
            database: Database name
        """
        conn = self.get_sql_connection(server, database)
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            logger.info("SQL statement executed successfully")
        
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to execute SQL: {str(e)}")
            raise
        
        finally:
            cursor.close()
            conn.close()
    
    def load_from_blob(self,
                      container_name: str,
                      blob_name: str,
                      file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from Azure Blob Storage
        
        Args:
            container_name: Blob container name
            blob_name: Blob file name
            file_type: File type ('csv', 'parquet', 'json')
        
        Returns:
            pandas DataFrame
        """
        connection_string = self.get_secret('AZURE_STORAGE_CONNECTION_STRING')
        
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        try:
            download_stream = blob_client.download_blob()
            data = download_stream.readall()
            
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(data))
            elif file_type == 'parquet':
                df = pd.read_parquet(io.BytesIO(data))
            elif file_type == 'json':
                df = pd.read_json(io.BytesIO(data))
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Loaded {len(df)} rows from blob: {blob_name}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load from blob: {str(e)}")
            raise
    
    def write_to_blob(self,
                     df: pd.DataFrame,
                     container_name: str,
                     blob_name: str,
                     file_type: str = 'csv'):
        """
        Write data to Azure Blob Storage
        
        Args:
            df: DataFrame to write
            container_name: Blob container name
            blob_name: Blob file name
            file_type: File type ('csv', 'parquet', 'json')
        """
        connection_string = self.get_secret('AZURE_STORAGE_CONNECTION_STRING')
        
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        try:
            # Convert DataFrame to bytes
            buffer = io.BytesIO()
            
            if file_type == 'csv':
                df.to_csv(buffer, index=False)
            elif file_type == 'parquet':
                df.to_parquet(buffer, index=False)
            elif file_type == 'json':
                df.to_json(buffer, orient='records')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            buffer.seek(0)
            
            # Upload to blob
            blob_client.upload_blob(buffer, overwrite=True)
            
            logger.info(f"Successfully wrote {len(df)} rows to blob: {blob_name}")
        
        except Exception as e:
            logger.error(f"Failed to write to blob: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize connector
    connector = AzureDataConnector(use_managed_identity=False)
    
    # Example 1: Load data from SQL
    query = """
        SELECT product_id, base_price, demand, inventory
        FROM products
        WHERE date >= ?
    """
    df = connector.load_data_from_sql(query, params=('2024-01-01',))
    print(f"Loaded {len(df)} products")
    
    # Example 2: Write predictions to SQL
    predictions_df = df.copy()
    predictions_df['predicted_price'] = predictions_df['base_price'] * 1.1
    predictions_df['prediction_date'] = pd.Timestamp.now()
    
    connector.write_predictions_to_sql(
        predictions_df[['product_id', 'predicted_price', 'prediction_date']],
        'price_predictions'
    )
    
    # Example 3: Load from Blob Storage
    blob_df = connector.load_from_blob(
        container_name='training-data',
        blob_name='historical_prices.csv'
    )
    print(f"Loaded {len(blob_df)} rows from blob")
