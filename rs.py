# Importing libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
# Reading the dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
print ("Datasets Loaded")
print ("Data Preprocessing Started")
final_dataset = ratings.pivot(index='userId',columns='movieId',values='rating')
final_dataset.fillna(0,inplace=True)
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[:, no_user_voted[no_user_voted > 10].index]
final_dataset = final_dataset.loc[no_movies_voted[no_movies_voted > 50].index, :]
def hot_encode(x): 
    if(x < 3.5): 
        return 0
    else: 
        return 1
final_dataset = final_dataset.applymap(hot_encode)
movieIdToName = dict()
for mid in final_dataset.columns:
    movieIdToName[mid] = movies[movies["movieId"] == mid]["title"].values[0]
finalLst = []
for i in final_dataset.index:
    lst = []
    for j in final_dataset.columns:
        if(final_dataset[j][i]):
            lst.append(j)
    finalLst.append(lst)
with open("dataset.txt", "w") as fp:
    for lst in finalLst:
        for x in lst:
            fp.write(str(x))
            fp.write(" ")
        fp.write("\n")
movieIdSize = 6
encoder = 100000
userCnt = 378
minSupport = 70
def generateKPlus1thSet(itemSet):
    length = len(itemSet)
    candidates = []   # all (k + 1) candidates    
    for (i, candidate) in enumerate(itemSet):
        for j in range(i + 1, length):
            nextCandidate = itemSet[j]
            if(candidate[:-movieIdSize] == nextCandidate[:-movieIdSize]):    
                newItem = candidate[:-movieIdSize] + candidate[-movieIdSize:] + nextCandidate[-movieIdSize:]
                candidates.append(newItem)            
    return candidates
def prune(Ck):
    Lk = []
    for item in Ck:
        if(Ck[item] >= minSupport):
            Lk.append(item)   
    return Lk
def calculateSupport(candidates):    
    Ck = dict()    
    for line in finalLst:
        line = list(map(lambda x: str(x + encoder), line))        
        for candidate in candidates:            
            if(candidate not in Ck):
                Ck[candidate] = 0               
            present = True            
            for k in range(0, len(candidate), movieIdSize):
                item = candidate[k: k + movieIdSize]                
                if(item not in line):
                    present = False
                    break                   
            if(present):
                Ck[candidate] += 1               
    return Ck
C1 = dict()
for line in finalLst:
    for item in line:
        item = str(item + encoder)
        C1[item] = C1.get(item, 0) + 1                   
L1 = prune(C1)      
L = generateKPlus1thSet(L1)
k = 2
while(L != []):    
    C = calculateSupport(L)    
    frequentItemset = prune(C)
    L = generateKPlus1thSet(frequentItemset)    
    k += 1
def decoder(frequentItemset):    
    y = [[itemSet[x : x + movieIdSize] for x in range(0, len(itemSet), movieIdSize)] for itemSet in frequentItemset]
    x1 = [list(map(lambda x: str(int(x) - encoder), z)) for z in y]    
    movieItemSet = []    
    for itemSet in x1:
        tempSet = []
        for movieId in itemSet:
            tempSet.append(movieIdToName[int(movieId)])           
        movieItemSet.append(tempSet)
    return movieItemSet
frequentItems = decoder(frequentItemset)
for itemSet in frequentItems:   
    for movie in itemSet:
        pass
freqItems = []
items = "".join(frequentItemset)
for k in range(0, len(items), movieIdSize):
    item = items[k: k + movieIdSize]
    support = (C1[item] / userCnt)
    movieName = frozenset([movieIdToName[int(item) - encoder]])
    freqItems.append([support, movieName])
freqDf = pd.DataFrame(freqItems, columns=["support", "itemsets"])
rules = association_rules(freqDf, metric ="confidence", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
final_dataset.columns = [movieIdToName[mid] for mid in final_dataset.columns]
frq_items = apriori(final_dataset, min_support = 0.3, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
for i in rules.index:
    antecedents = []
    consequents = []
    for j in rules.antecedents[i]:
        antecedents.append(j[:30])
    for k in rules.consequents[i]:
        consequents.append(k[:30])
print ("Data Preprocessing Finished")


def getRecommendation(movie):
    similarMovies = []
    for movies in frequentItems:
        if movie in movies:
            similarMovies.extend(movies)
    return similarMovies

print('\n\n','Enter Movie Name :')
movie = input()
print("The Recommended Movies are\n")
recommended_movies = getRecommendation(movie)
for movies in recommended_movies:
    if(movies != movie):
        print(movies)

while True:
  print('\n\n','Enter Movie Name :')
  movie = input()
  print("The Recommended Movies are\n")
  recommended_movies = getRecommendation(movie)
  for movies in recommended_movies:
         if(movies != movie):
             print(movies)
  print('=======================')
