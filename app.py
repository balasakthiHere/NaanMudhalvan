from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        data = pd.read_csv('house_data.csv')
        data = pd.get_dummies(data, columns=['Location', 'Zip_Code'])

        X = data.drop('Price', axis=1)
        y = data['Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': predictions})

        mse = mean_squared_error(y_test, predictions)
        rmse = (mse) ** 0.5

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.title('Actual Prices vs. Predicted Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('result.html', mse=rmse, plot_url=plot_url)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
