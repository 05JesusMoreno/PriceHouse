from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_Casas.pkl')
scaler = joblib.load('escalador (3).pkl')
app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        OverallQual = float(request.form['OverallQual'])
        GrLivArea = float(request.form['GrLivArea'])
        secondFlrSF = float(request.form['2ndFlrSF'])
        TotalBsmtSF = float(request.form['TotalBsmtSF'])
        BsmtFinSF1 = float(request.form['BsmtFinSF1'])
        firstFlrSF = float(request.form['1stFlrSF'])
        Neighborhood = float(request.form['Neighborhood'])
        KitchenQual = float(request.form['KitchenQual'])
        GarageCars = float(request.form['GarageCars'])
       
        # Crear DataFrame
        data_df = pd.DataFrame([[OverallQual, GrLivArea, secondFlrSF, TotalBsmtSF, BsmtFinSF1, firstFlrSF,Neighborhood,KitchenQual,GarageCars]],
                               columns=['OverallQual', 'GrLivArea', '2ndFlrSF', 'TotalBsmtSF', 'BsmtFinSF1', '1stFlrSF','Neighborhood','KitchenQual','GarageCars'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Escalar los datos de entrada
        data_scaled = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {data_scaled}')

        # Realizar predicciones
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
        
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
