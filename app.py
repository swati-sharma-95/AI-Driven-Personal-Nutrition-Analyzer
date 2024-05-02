from flask import Flask, render_template, url_for, request

app = Flask(__name__)

 

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/table',methods=['GET'])
def table():
    return render_template("table.html")



@app.route('/result',methods=['POST', 'GET'])
def result():
    file_path = '/Users/megh/Downloads/image.jpeg'
    import tensorflow.compat.v2 as tf
    import tensorflow_hub as hub
    import numpy as np
    import cv2
    from skimage import io
    import pandas as pd
    import requests
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl import load_workbook
    from datetime import date
    from scipy.optimize import linprog


    # TF2 version
    m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

    # Read and preprocess the image from Google Drive
    input_shape = (224, 224)
    image = np.asarray(io.imread(file_path), dtype="float")
    image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
    image = image / image.max()
    images = np.expand_dims(image, 0)

    # Predict using the model
    output = m(images)
    predicted_index = output.numpy().argmax()

    # Load label map
    labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
    classes = list(pd.read_csv(labelmap_url)["name"])

    # Print the prediction
    print("Prediction: ", classes[predicted_index])
    name = classes[predicted_index]
  
    height = float(request.form.get("height"))
    weight =float(request.form.get("weight"))
    age=float(request.form.get("age"))
    gender=request.form.get("gender")
    activity = float(request.form.get("activity"))
    username =request.form.get("username")
    serving =float(request.form.get("serving"))

    API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
    API_KEY = "CN1KSkobrGbszhyPOla55aEDrbFo9ezYnpcWy05E" 
    params = {"query": name, "pageSize": 1}
    headers = {"x-api-key": API_KEY}

    response = requests.get(API_URL, params=params, headers=headers)

    food = response.json()["foods"][0]
    nutrients = food["foodNutrients"]
    nutrients_dict = {}
    for nutrient in nutrients:
        nutrients_dict[nutrient["nutrientName"]] = nutrient["value"]

    def calculate_bmi(weight_kg, height_m):
        bmi = weight_kg / (height_m ** 2)
        return bmi

    def calculate_bmr(weight_kg, height_cm, age, gender):
        if gender.lower() == "male":
            bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        elif gender.lower() == "female":
            bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        else:
            raise ValueError("Invalid gender specified")
        return bmr

    def calculate_daily_calories(bmr, activity_factor):
        daily_calories = bmr * activity_factor
        return daily_calories
    
    def calculate_macronutrient_calories(carbohydrates, proteins, fats):
        carb_calories = carbohydrates * 4  # 4 calories per gram
        protein_calories = proteins * 4    # 4 calories per gram
        fat_calories = fats * 9           # 9 calories per gram
        return carb_calories, protein_calories, fat_calories


    date_of_entry = date.today()
    bmr = calculate_bmr(weight,height,age,gender)
    bmi= calculate_bmi(weight,height)
    daily_calories = calculate_daily_calories(bmr,activity)
    daily_carbs_gms = .30*daily_calories/4
    daily_protein_gms= 0.4*daily_calories/4
    daily_fat_gms = 0.3*daily_calories/9
         
    food_carbs_gms = abs(nutrients_dict.get("Carbohydrate, by difference", 0) * 4 *serving)
    food_protein_gms = abs(nutrients_dict.get("Protein", 0) * 4  * serving)
    food_fat_gms = abs(nutrients_dict.get("Total lipid (fat)", 0) * 9 * serving)
    food_total_calories = food_carbs_gms+food_protein_gms+food_fat_gms
    carbs_remaining= daily_carbs_gms-food_carbs_gms
    fats_remaining= daily_fat_gms-food_fat_gms
    protein_remaining=daily_protein_gms-food_protein_gms
    total_cals_remaining = daily_calories-food_total_calories
 
    obj_coefficients = [-1, -1,-1,-1,-1]  # Objective: maximize servings (negated to convert from minimization to maximization)

    # Coefficients of the inequality constraints (carbs, fats, proteins)
    # 1 serving Eggs Calories: 72. Protein: 6 grams. Fat: 5 grams. Carbs: 1 gram.
    # 1 cup vegetables Calories: 60 Protein: 2.6 gms Fat: 0.1 gms Carbs : 12 gms
    # 1 cup chicken: Caloties : 230 Protein: 43gms Fat: 5gms Carbs: 1gm
    # 1 cup fruits: Calories: 97 Protein: 1.4gms Fat0.5gms and Carbs: 24gms
    # 1 cup rice: Calories: 195 Protein: 4.6gms Fat 0.6gms Carbs: 41gms

    lhs_coefficients = [
        [72,69,230,97,195], # Calories constraint coefficients for All dishes
        [1,12,1,24,41],    # Carbs constraint coefficients for All dishes
        [5,0.1,5,0.5,0.6],    # Fats constraint coefficients for All dishes
        [26,2.6,43,1.4,4.6]     # Proteins constraint coefficients for All dishes
    ]

    # RHS values of the inequality constraints (available carbs, fats, proteins)
    rhs_values = [total_cals_remaining, carbs_remaining, fats_remaining,protein_remaining]  # Available calories, carbs, fats, proteins

    # Bounds for the variables (servings of Dish 1 and Dish 2)
    bounds = [(0, None), (0, None),(0, None),(0, None),(0, None)]  # Non-negative servings

    # Solve the linear programming problem (maximization)
    result = linprog(c=obj_coefficients, A_ub=lhs_coefficients, b_ub=rhs_values, bounds=bounds, method='highs')

    if result.success:
        servings_dish1 = result.x[0]
        servings_dish2 = result.x[1]
        servings_dish3 = result.x[2]
        servings_dish4 = result.x[3]
        servings_dish5 = result.x[4]

        from openai import OpenAI
        client = OpenAI(api_key='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') 
        # Formulate a query for OpenAI
        query = f"give me dinner and snacks single servings with Maximum Servings of eggs: {servings_dish1:.2f} and Maximum Servings of vegetables : {servings_dish2:.2f} cups and Maximum Servings of chicken : {servings_dish3:.2f} cups and Maximum Servings of fruits : {servings_dish4:.2f} cups and Maximum Servings of rice : {servings_dish5:.2f} cups so i can meet my nutritional goals for today. It can be multiple dishes but only 1 serving per dish.I just need names of the dishes with no recipies."

        # Call OpenAI to get recommendations
        response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=query,
        max_tokens=300
        )
         # Extract and display the model's response
        model_response = response.choices[0].text
        recommendation = model_response
    else:
       recommendation = "No optimal recommendations found"
    


    wb_append = load_workbook("/Users/megh/Desktop/BAN693-CapstoneProject/Demo3/Data.xlsx")
    sheet = wb_append.active
    data = (date_of_entry,username,height,weight,age,gender,bmi,bmr,activity,daily_calories,daily_carbs_gms,daily_protein_gms,daily_fat_gms,name,food_total_calories,food_carbs_gms,food_protein_gms,food_fat_gms,carbs_remaining,fats_remaining,protein_remaining,total_cals_remaining)
    sheet.append(data)
    wb_append.save("/Users/megh/Desktop/BAN693-CapstoneProject/Demo3/Data.xlsx")
    
    df = pd.read_excel("/Users/megh/Desktop/BAN693-CapstoneProject/Demo3/Data.xlsx")
    df= df.loc[df['Name'] == username]
    result = df.to_html(classes = 'table table-stripped')

    text_file = open("./templates/table.html", "w") 
    text_file.write(result) 
    text_file.close() 
    
    return render_template('index.html', name = name, recommendation =recommendation)
    




if __name__ == "__main__":
    app.run(debug=True,port=5005)
