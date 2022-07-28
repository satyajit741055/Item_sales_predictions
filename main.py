from flask import Flask, request
from sales.logger import logging,get_log_dataframe
from sales.config.configuration import Configuration
from sales.constant import CONFIG_DIR, get_current_time_stamp
from sales.pipeline.pipeline import Pipeline
from sales.entity.sales_predictor import salesPredictor, SalesData
from flask import send_file, abort, render_template,send_from_directory
import os





ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "sales"
SAVED_MODELS_DIR_NAME = "saved_models"
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)




SALES_DATA_KEY = "Sales_data"
ITEM_OUTLET_SALES_KEY = "Item_Outlet_Sales"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/artifact', defaults={'req_path':'sales'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("sales", exist_ok=True)
    # Joining the base and the requested path 
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path) 
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    logging.info(f"artifact results : {result}")
    return render_template('files.html', result=result)


# Needs to check 
@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        SALES_DATA_KEY: None,
        ITEM_OUTLET_SALES_KEY: None
    }

    if request.method == 'POST':
        Item_Identifier = str(request.form['Item_Identifier'])
        Item_Weight = float(request.form['Item_Weight'])
        Item_Fat_Content = str(request.form['Item_Fat_Content'])
        Item_Visibility = float(request.form['Item_Visibility'])
        Item_Type = str(request.form['Item_Type'])
        Item_MRP = float(request.form['Item_MRP'])
        Outlet_Establishment_Year = float(request.form['Outlet_Establishment_Year'])
        Outlet_Size = str(request.form['Outlet_Size'])
        Outlet_Location_Type = str(request.form['Outlet_Location_Type'])
        Outlet_Type =  str(request.form['Outlet_Type'])
        Item_Outlet_Sales =  None
        Outlet_Identifier =  str(request.form['Outlet_Identifier'])



        sales_data = SalesData(Item_Identifier=Item_Identifier,
                                   Item_Weight=Item_Weight,
                                   Item_Fat_Content=Item_Fat_Content,
                                   Item_Visibility=Item_Visibility,
                                   Item_Type=Item_Type,
                                   Item_MRP=Item_MRP,
                                   Outlet_Establishment_Year=Outlet_Establishment_Year,
                                   Outlet_Size=Outlet_Size,
                                   Outlet_Location_Type=Outlet_Location_Type,
                                   Outlet_Type = Outlet_Type,
                                   Item_Outlet_Sales = Item_Outlet_Sales, 
                                   Outlet_Identifier = Outlet_Identifier
                                   )
        sales_df = sales_data.get_sales_input_data_frame()
        sales_predictor = salesPredictor(model_dir=MODEL_DIR)
        Item_Outlet_Sales = sales_predictor.predict(X=sales_df)
        context = {
            SALES_DATA_KEY: sales_data.get_sales_data_as_dict(),
            ITEM_OUTLET_SALES_KEY: Item_Outlet_Sales,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)

@app.route('/saved_models', defaults={'req_path':'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    logging.info(f"saved templets results : {result}")
    return render_template('saved_model_files.html', result=result)

@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    logging.info(f"logs results : {result}")
    return render_template('log_files.html', result=result)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)



if __name__ == "__main__":
    app.run(debug=True)