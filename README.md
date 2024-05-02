# AI-Driven-Personal-Nutrition-Analyzer

The AI-Driven Personal Nutrition Analyzer is a cutting-edge web application that empowers users to monitor and analyze their dietary habits through a seamless integration of image recognition, nutritional analysis, and personalized recommendations. This capstone project aims to revolutionize the way individuals approach their nutritional needs, providing a comprehensive and user-friendly solution.

## Key Features:

- Image Recognition: Leveraging deep learning models (TensorFlow, OpenCV) to accurately identify and classify food items from user-uploaded images.
- Nutritional Data Extraction: Integration with the USDA FoodData Central API to retrieve detailed nutritional information for recognized food items.
- Personalized Calculations: Computation of personalized metrics (BMI, BMR, daily caloric needs) based on user attributes (height, weight, age, gender, activity level).
- Dietary Optimization: Utilization of linear programming techniques (SciPy's linprog solver) to suggest optimized meal plans and healthier food alternatives.
- Recommendation Generation: Integration with OpenAI's GPT-3 API to generate context-aware food recommendations tailored to individual preferences and constraints.
- Data Visualization and Tracking: User-friendly charts and tables for visualizing nutritional intake, progress towards goals, and longitudinal tracking of dietary patterns.

## Technologies Used:

- Python
- TensorFlow, OpenCV (Image Recognition)
- USDA FoodData Central API (Nutritional Data)
- Flask (Web Application Framework)
- SciPy (Linear Programming Optimization)
- OpenAI GPT-3 API (Recommendation Generation)
- Pandas, openpyxl (Data Manipulation and Storage)

## Getting Started:
To run the AI-Driven Personal Nutrition Analyzer locally, follow these steps:

1. Clone the repository: git clone https://github.com/your-username/personal-nutrition-analyzer.git
2. Install the required dependencies: pip install -r requirements.txt
3. Obtain API keys for USDA FoodData Central and OpenAI GPT-3
4. Configure the API keys in the appropriate configuration files
5. Run the Flask application: python app.py
6. Access the application through your web browser at http://localhost:5000

## Contributions:
Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments:
This project was developed as part of the BAN 693-01 Capstone course at [University Name] under the guidance of Professor Surendra Sarnikar. We would like to express our gratitude to our professor and teammates (Megh Dave, Akhil Mohan, and Pruthvi Aala) for their invaluable support and contributions.
