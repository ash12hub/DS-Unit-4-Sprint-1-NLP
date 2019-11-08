from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

LOCAL_TRAIN = './data/train.csv'
LOCAL_TEXT = './data/test.csv'
training_df = pd.read_csv(LOCAL_TRAIN)
testing_df = pd.read_csv(LOCAL_TEST)

x_train = training_df['description']
y_train = training_df['category']

x_test = testing_df['description']

vect = TfidfVectorizer(stop_words='english')
rfc = RandomForestClassifier()

pipe = Pipeline([
                ('vect', vect),
                ('clf', rfc)
               ])

parameters = {
    'vect__max_df': (0.75, 1.0),
    'vect__min_df': (.02, .07),
    'clf__n_estimators': (5, 10),
    'clf__max_depth': (20, 40)
}

grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1,
                           verbose=1)
grid_search.fit(x_train, y_train)
predictions = grid_search.predict(x_test)

print('best score:', grid_search.best_score_)


def generate_csv(grid_search):
    submission_df = pd.DataFrame({'id': testing_df['id'], 'category': predictions})
    submission_df['category'] = submission_df['category'].astype('int64')
    submission_df.to_csv('whiskey_submission.csv', index=False)
    return submission_df


generate_csv(grid_search)
