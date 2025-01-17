# Part1
## (b) 
![image](https://github.com/user-attachments/assets/3f586662-7955-4351-892c-66206536c4da)  
Learning curve:  
![image](https://github.com/user-attachments/assets/3b1eeb16-1661-4d97-b086-a0f1b0ba16d1)  
Training predicts:  
![image](https://github.com/user-attachments/assets/b4045c56-0096-4973-88fc-0df305b76dc7)  
Test predicts:  
![image](https://github.com/user-attachments/assets/8a2a4d09-fd0d-4306-a8c2-2a6d637ef8ea)  
在這邊我有先利用normalization使資料更好訓練，且最後發現此次作業資料量其實不大，直接使用整個training set來做訓練即可，不用mini batch可在Test ERMS上效果更好。  
![image](https://github.com/user-attachments/assets/9940c073-177a-4fcf-8550-ed9a80e18f63)  
## (c)
![image](https://github.com/user-attachments/assets/b8e23bc1-b61a-4776-88a9-ee2477d0a975)  
Features:  
0.	Relative Compactness  
1.	Surface Area  
2.	Wall Area  
3.	Roof Area  
4.	Overall Height  
5.	Orientation  
6.	Glazing Area  
7.	Glazing Area Distribution  
![image](https://github.com/user-attachments/assets/dd391977-adad-4507-a018-26733495d7a0)  
![image](https://github.com/user-attachments/assets/eecbfe79-21d6-4a02-9524-0fc495b74a74)  
由上表可知Orientation, Glazing Area, Glazing Area Distribution 三個features是對 Heating Load 最無關的，其中Orientation,  Glazing Area Distribution 是跟位置有關，但隨著位置改動並不太會對Heating Load 有著決定性的改變， Glazing Area本身數值也只有四種，且與 Heating Load 的變化無直接關係，相較之下 Overall Height 雖然數值只有兩種，但與 Heating Load 大略呈正相關。  
而Wall Area 則比較尷尬，有些數據似乎呈正相關，有些似乎無關，沒有Features 0,1,3,4 那麼的決定性。  
由下圖更可得知 Features 0,1,3,4 皆大略可看出相關性:  
Features 0,1,3,4:  
![image](https://github.com/user-attachments/assets/7df0be62-89b9-430e-8a44-6998779d64ef)  
Feature 2:  
![image](https://github.com/user-attachments/assets/ca1cab9b-b450-4a57-8e57-b21cf5b723c2)  
# Part 2:
## (b)
![image](https://github.com/user-attachments/assets/feb37f9c-9e11-4928-9e47-05e220641768)  
Learning curve:  
![image](https://github.com/user-attachments/assets/5f92067b-2f0d-433f-9a02-864be3d96379)  
與 Part 1 相同，先利用normalization使資料更好訓練，並直接使用整個training set來做訓練即可。  
![image](https://github.com/user-attachments/assets/26f3f0a1-dfaa-482a-b756-ff63b4e1fdfc)  
## (c)
![image](https://github.com/user-attachments/assets/f38b3ac8-ddf5-4c89-a5fc-b3f14e5360a4)
![image](https://github.com/user-attachments/assets/998cd521-bc77-4e98-ba31-d191f56d14ee)  
(1)	epoch = 50且output 前一層的nodes 數變小時:  
可看到點狀圖散布範圍大略成越來越大的趨勢，如:nodes 64的圖片約為5*5的大小；nodes 4 則為7*8的大小，推論為當node數量太少時，他會偏頗於其中幾個數據，使這些局部數據的cross-entropy下降，而當node變多時，則會找到一個更全面性的參數，使cross-entropy 全面性的下降。  
(2)	epoch = 1000且output 前一層的nodes 數變小時:  
由於node 數小的仍然無法全面性的照顧到每個數據，在圖上分布較為發散，未成一個明顯的直線，且分類效果也較不突出(註1)，而在node越多的情況下，數據分布較為集中，且分類效果越好，可全面性的照顧每個數據。  

註1:  
這裡的分類應看的是x軸的值與y軸的值的大小，如想分類為class 1，其x軸的值應比y軸的值大越多越好，class 0則反之，因此如果分布成通過原點的負斜率直線，只要數據在直線上且看x軸(或y軸)是否大於零即可分類，而node數小的較難看出這種直線分布，分類能力較弱。  



