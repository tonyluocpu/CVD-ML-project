
import pandas as pd
 
'''
all_three_dataset.npynb

wikicounty.csv: take from Land Area (km2) to longitude, merge on "fips_county"
immigration_cleaned.csv: merge on "fips_county"
original_dataset.csv: Merge on "statecounty_x"

'''

wiki, immi, ori = pd.read_csv("wikicounty.csv"), pd.read_csv("immigration_cleaned.csv"), pd.read_csv("original_dataset.csv")

print(wiki.columns)
print(immi.columns)
print(wiki.shape, immi.shape, ori.shape)

print(wiki["FIPS"])
print("________")
print(immi["stname"])
print("________")
print(ori["statecounty_x"])

