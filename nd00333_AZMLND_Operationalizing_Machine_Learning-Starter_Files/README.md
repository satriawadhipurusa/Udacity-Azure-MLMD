# Operationalizing Machine Learning Pipeline

In this project we created a Pipeline of Automated Machine Learning that utilizes many features of Azure Machine Learning Studio and SDK. The model will be trained using Automated ML, which is an optimization algorithm that will aim to reach the best metrics or lowest error through the combinations of:

- Preprocessing steps - the algorithm will try to combine preprocessing steps such as StandardScaler, MaxAbsScaler, etc
- Classifier / Regressor Model - the algorthm will search the best alglgorithm from the likes of Logistic Regression, XGBoost, Random Forest, etc
- Hyperparameters of each Model - the hyperparams of each model will included in the search space

The Automated ML help to kickstart the project and make iterations faster, so we can put something decent into production. While the pipeline itself made, so the whole experiment can be made **reproducible**. All of them are tracked and versioned in Azure ML and make data debugging or model debugging easier by tracking data Lineage or data Provenance.

In the end, the model will be consumed as REST HTTP Service endpoint, and not deployed to mobile or edge devices. The advantage is that we can scale the model using Inference Cluster in the Cloud or using accelerator like GPU. The downside is, this won't be available without internet.

## Architectural Diagram

![udacity-c2-arch](assets/udacity-c2-arch.png)
On high overview, this project utilizes the following components on Azure Cloud:

- Service Principal
- Azure Machine Learning Studio
- Azure Container Instances
- Azure Storage Services

First, the datasets must be uploaded and registered to Azure Datasets. This will be consumed as the entrypoint of the Machine Learning Pipeline, that will be executed by Compute Cluster. The steps only consist of Automated ML, that will determine the best model. Once the training completed, the model will be registered on Azure Machine Learning Studio's model registry. We can deploy this model on Azure Container Instances (ACI) with authentication enabled by default. Additionally, Application Insight can also be enabled for monitoring and logging. This model endpoint can be used for inference by clients like mobile or webapp by real-time or batch. All of this made possible by Service Principal that has been shared the Workspace's ownership role.

## Future Improvement

In the future, the followings can be done to improve the project further:

- Create an end-to-end Pipeline from data ingestion, transformation, and deployment without manual interference
- Set a recurring scheduled or triggered scheduled based on alarms such as Data Drift or Model metrics drift
- Deploy to kubernetes and make canary release for new model versions so A/B test can be done before switch completely to the new one
- Set a monitoring of important metrics not just system metrics from Application Insights but also Model Metrics and Business Metrics
- Make parameter to be dynamic, such as user can choose data source or features to be included (i.e. integrate Feature Store into the pipeline)

## Key Steps

1. **Create and configure Service Principal**
   Service principal must be created and configured to use Workspace, so authentication can be done automatically

Make sure the azure sdk and cli are installed then

```sh
$ az ad sp create-for-rbac --sdk-auth --name ml-auth
```

![service-principal](assets/service-principal.png)

we can show the id has been created by checking the `clientId` created from the above command

```
$ az ad sp show --id <yourClientId>
```

then we execute this command instead, since v2 of azure ml cli is used

```
az role assignment create --assignee <objectId> --role "Owner" --scope "/<subscriptionId>/resourceGroups/<yourresourceName>/providers/Microsoft.MachineLearningServices/workspaces/<yourWorkspace>"
```

![role-assignment](assets/role-assignment.png)

2. **Register Dataset & Create Automated ML Run**
   Pre-requisites:

- [`bankmarketing-train.csv`](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) dataset is created and registered using the Datasets Tab
  ![dataset](assets/dataset.png)
- AML Compute Cluster is created and set to use `Standard DS12_v2` with minimum of 1 node

After that, create an Automated ML run using the compute cluster and the registered dataset, choose 1 hour as exit criterion and set the _concurrency_ number according to the number of compute cluster. Once in a while, check the experiments tab to see if the Automated ML run has finished.

Finished Automated ML:
![finished-automlrun](assets/finished-automlrun.png)

Best Model:
![best-model](assets/best-model.png)
![best-model-summary](assets/best-model-summary.png)

3. **Deploy the Best Model & Enable Logging**
   The best model can be deploy by clicking the Deploy tab, and choose "Deploy to webservice". Next choose the Azure Container Instances (ACI) deployment with authentication enabled.

After deployment finished, application insights can be enabled using Azure ML Python SDK, and also print out the logging. The script is available in `logging.py`

Enabled App Insights:
![app-insights-enabled](assets/app-insights-enabled.png)

Logs Output:
![logs-output](assets/logs-output.png)

4. **Interact with Swagger**
   Copy or download the swagger from the Endpoints page of deployed model. We then use the `swagger.sh1` and `serve.py` to interact with the swagger and see how we can send the payload to the prediction endpoint using the HTTP API. The swagger should run in http://localhost (port 80) and point the location of `swagger.json` in http://localhost:8000/swagger.json (must be in the same directory!)

![swagger](assets/swagger.png)
![swagger2](assets/swagger2.png)
![swagger3](assets/swagger3.png)

5. **Consume the Endpoint**
   The model endpoint can be consume using the prediction URI and api key that we get from the Endpoint page of deployed model. We then use the `endpoint.py` by input the needed variable [redacted] and run the script to send some data.

   Prediction Output:
   ![endpoint](assets/endpoint.png)

   Log of Prediction:
   ![logs-prediction](assets/logs-prediction.png)

   You can also benchmark your endpoint to see some stats like Transaction per Second (TPS) or average response time (ms), this can then be used as a baseline of your model endpoints behavior in day to day operations after deployment. Use the `benchmark.sh` script and input the needed variable like prediction URI [redacted] and api key [redacted].

   ![benchmark](assets/benchmark.png)
   ![benchmark2](assets/benchmark2.png)

6. **Create and Publish the Pipeline**
   Once the POC has completed, we can then write the pipeline to make it reproducible and published it so people can use and trigger our pipeline using HTTP API. First we use `aml-pipelines-with-automated-machine-learning-step.ipynb` and upload it to AML Compute Instance and interact with Azure Machine Learning using the SDK.

   Pipeline Created:
   ![pipeline](assets/pipeline.png)

   Pipeline Modules:
   ![pipeline-module](assets/pipeline-module.png)

   Published Pipeline:
   ![published-pipeline-overview](assets/published-pipeline-overview.png)

7. **Interact with the Published Pipeline**
   We can then interact with pipeline endpoint using any HTTP tools, we will use python `requests` module for this purpose. In the same notebook, we send an http request and scheduled the pipeline to run. Since the pipeline are using the same parameter, hence the result will be the same (using the cached previous run)

   Pipeline Widget in Notebook:
   ![pipeline-widget](assets/pipeline-widget.png)

   Pipeline Endpoint:
   ![pipeline-endpoint](assets/pipeline-endpoint.png)

   Pipeline Scheduled Run:
   ![scheduled-pipeline](assets/scheduled-pipeline.png)

## Screen Recording

To access the full screen recording visit the youtube link: [Udacity - Azure MLND - Operationalizing Maching Learning](https://www.youtube.com/watch?v=P83NT-55z4U)
