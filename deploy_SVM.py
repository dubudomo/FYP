import pickle
import streamlit as st
import pandas as pd
def predict(sentence):
    
    pred_res=[]
    
    with open('vectorizer1.pickle', 'rb') as handle:
        vectorizer1 = pickle.load(handle)
    with open('model1.pickle', 'rb') as handle:
        model1 = pickle.load(handle)
    pred1 = model1.predict(vectorizer1.transform([sentence]))

    pred_res.append(pred1)

    with open('vectorizer2.pickle', 'rb') as handle:
        vectorizer2 = pickle.load(handle)
    with open('model2.pickle', 'rb') as handle:
        model2 = pickle.load(handle)
    pred2 = model2.predict(vectorizer2.transform([sentence]))

    pred_res.append(pred2)
    
    with open('vectorizer3.pickle', 'rb') as handle:
        vectorizer3 = pickle.load(handle)
    with open('model3.pickle', 'rb') as handle:
        model3 = pickle.load(handle)
    pred3 = model3.predict(vectorizer3.transform([sentence]))

    pred_res.append(pred3)
    
    with open('vectorizer4.pickle', 'rb') as handle:
        vectorizer4 = pickle.load(handle)
    with open('model4.pickle', 'rb') as handle:
        model4 = pickle.load(handle)
    pred4 = model4.predict(vectorizer4.transform([sentence]))

    pred_res.append(pred4)
    
    with open('vectorizer5.pickle', 'rb') as handle:
        vectorizer5 = pickle.load(handle)
    with open('model5.pickle', 'rb') as handle:
        model5 = pickle.load(handle)
    pred5 = model5.predict(vectorizer5.transform([sentence]))

    pred_res.append(pred5)
    pred_df = pd.DataFrame(pred_res, columns=['sentiment'])
    for i in range(len(pred_df)):
        if (pred_df['sentiment'].iloc[i]==int(2)):
            pred_df['sentiment'].iloc[i] = "positive"
        elif (pred_df['sentiment'].iloc[i]==int(1)):
            pred_df['sentiment'].iloc[i] = "neutral"
        else:
            pred_df['sentiment'].iloc[i] = "negative"
    return pred_df


st.write("Please input a sentence here")
user_input = st.text_area("Hit CTRL+Enter to run", "This is an amazing product")
df = predict(user_input)
st.title("Sentiment Analysis in Product Review")
st.write("The final output sentiment")
st.dataframe(df['sentiment'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
)

