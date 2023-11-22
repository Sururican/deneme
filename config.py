import os

settings = {
    'host': os.environ.get('ACCOUNT_HOST', 'https://databasetrade.documents.azure.com:443/'),
    'master_key': os.environ.get('ACCOUNT_KEY', 'L5ERjHsiZ4Dp3dSMSiRNofZ6budhqYXnRvbDShEo9yw669ogcWHm91ORfkMinDIlGVVjUHIT4PRIACDbmqtPtw=='),
    'database_id': os.environ.get('COSMOS_DATABASE', 'ToDoList'),
    'container_id': os.environ.get('COSMOS_CONTAINER', 'Items'),
}