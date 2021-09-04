import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
toefl = df["TOEFL Score"].tolist()
gre = df["GRE Score"].tolist()
colours = []


score = df[["GRE Score", "TOEFL Score"]]
chances = df["Chance of admit"]

score_train, score_test, chances_train, chances_test = train_test_split(score, chances, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train, chances_train)

chancesprediction = classifier.predict(score_test)
print("Accuracy: ", accuracy_score(chances_test, chancesprediction))

sc_x = StandardScaler()
score_train = sc_x.fit_transform(score_train)
usertoefl = int(input("Enter your TOEFL score: "))
usergre = int(input("Enter your GRE Score: "))

usertest = sc_x.transform([[usergre, usertoefl]])
userchancesprediction = classifier.predict(usertest)
if userchancesprediction[0]==1:
  print("The user may not pass")
else:
  print("The user may pass")

for data in chances:
  if data == 1:
    colours.append("green")
  else:
    colours.append("red")

fig = go.Figure(data = go.Scatter(
    x = toefl,
    y = gre,
    mode = "markers",
    marker = dict(color = colours)
))
fig.show()