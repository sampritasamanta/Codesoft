# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
data = pd.read_csv('IMDb Movies India.csv')

# Preprocessing
# Drop irrelevant columns
data = data.drop(columns=['imdb_title_id', 'title', 'date_published', 'language', 'writer', 'description', 'budget', 'usa_gross_income', 'worlwide_gross_income', 'metascore', 'reviews_from_users', 'reviews_from_critics'])

# Handling missing values
data = data.dropna()

# Feature Engineering
# Convert categorical variables to numerical using Label Encoding
label_encoders = {}
for col in ['genre', 'director', 'actors']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features and target
X = data.drop(columns=['avg_vote'])
y = data['avg_vote']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for preprocessing and modeling
# Numeric features
numeric_features = ['duration', 'year', 'votes']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Categorical features
categorical_features = ['genre', 'director', 'actors']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append regression model to preprocessing steps
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
example_movie = {
    'duration': 120,
    'year': 2000,
    'genre': 'Action',
    'director': 'Christopher Nolan',
    'actors': 'Tom Hanks',
    'votes': 10000
}

example_movie_df = pd.DataFrame(example_movie, index=[0])
example_movie_df['genre'] = label_encoders['genre'].transform(example_movie_df['genre'])
example_movie_df['director'] = label_encoders['director'].transform(example_movie_df['director'])
example_movie_df['actors'] = label_encoders['actors'].transform(example_movie_df['actors'])

predicted_rating = model.predict(example_movie_df)
print(f'Predicted Rating for the example movie: {predicted_rating[0]}')
