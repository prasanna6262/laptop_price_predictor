import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
    
def load_and_preprocess_data(csv_path):
    df = pd.read_csv('data/laptops.csv')
# Handle missing values
    df=df.dropna()

    label_encoders={}
    for col in ['Brand','Model Name','Processor','OS','GPU']:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
        label_encoders[col]=le
# Feature target split
    X= df.drop('Price',axis=1)
    y= df['Price']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test, label_encoders

