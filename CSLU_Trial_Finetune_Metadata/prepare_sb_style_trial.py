with open("trial_list_combined_with_age_gender_abs") as f:
    rws = f.readlines()

i=0
for rw in rws:
  if i == 0:
    i = i+1
  else:
    rw = rw.strip()
    uttid1 = rw.split(",")[0].split("/")[-1]
    spkid1 =  rw.split(",")[0].split("/")[-2]
    uttid2 = rw.split(",")[1].split("/")[-1]
    spkid2 = rw.split(",")[1].split("/")[-2]
    type1 = rw.split(",")[2]
    type2 = rw.split(",")[3]
    target = rw.split(",")[4]
    age1 = rw.split(",")[5]
    age2 = rw.split(",")[6]
    gender1 = rw.split(",")[7]
    gender2 = rw.split(",")[8]
    print(spkid1 + "/1/" + uttid1 , spkid2 + "/1/" + uttid2, type1,type2,target,age1, age2,gender1,gender2,sep=",")
