# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:34:55 2022

@author: rvamsikrishna
"""

#%%
#%%
#%%

# Python program to find smallest number in a list

list1 = [10, 20, 4, 45, 99]

# sorting the list
list1.sort()

# printing the first element
print("Smallest element is:", *list1[:1])

print("Smallest element is:", min(list1))

print("Largest element is:", list1[-1])

print("Largest element is:", max(list1))


#%%
# Python program to find smallest number in a list

# creating empty list
list1 = []

# asking number of elements to put in list
num = int(input("Enter number of elements in list: "))

# iterating till num to append elements in list
for i in range(1, num + 1):
	ele= int(input("Enter elements: "))
	list1.append(ele)
	
# print maximum element
print("Smallest element is:", min(list1))

print("Largest element is:", max(list1))

#%%
# Python program to find smallest number in a list

l=[ int(l) for l in input("List:").split(",")]

print("The list is ",l)

# Assign first element as a minimum.
min1 = l[0]

for i in range(len(l)):

	# If the other element is min than first element
	if l[i] < min1:
		min1 = l[i] #It will change

print("The smallest element in the list is ",min1)

#%%
# Python program to find largest number in a list

def myMax(list1):
	# Assume first number in list is largest initially and assign it to variable "max"
	max = list1[0]
	# Now traverse through the list and compareeach number with "max" value.
    #Whichever is largest assign that value to "max'.
	for x in list1:
		if x > max :
			max = x           
	return max


# Driver code
list1 = [10, 20, 4, 45, 99]
print("Largest element is:", myMax(list1))

#%%
#%%
#%%
#Python program to find N largest elements from a list

def Nmaxelements(list1, N):
	final_list = []
    
	for i in range(0, N):
		max1 = 0
		
		for j in range(len(list1)):	
			if list1[j] > max1:
				max1 = list1[j];
				
		list1.remove(max1);
		final_list.append(max1)
		
	print(final_list)

# Driver code
list1 = [2, 6, 41, 85, 0, 3, 7, 6, 10]
N = 2

# Calling the function
Nmaxelements(list1, N)

#%%
# Python program to find N largest element from given list of integers

l = [1000,298,3579,100,200,-45,900]
n = 4

l.sort()
print(l[-n:])

#%%
#%%
#%%
# Python program to print Even/Odd numbers in a list

#using for loop
#------------------

# list of numbers
list1 = [10, 21, 4, 45, 66, 93]

# iterating each number in list
for num in list1:
	# checking condition
	if num % 2 == 0: #if num % 2 != 0: for odd num in the list
    
		print(num, end=" ")

#%%
#Method 2: Using while loop 
#-------------------------------        
        
# list of numbers
list1 = [10, 24, 4, 45, 66, 93]
num = 0

# using while loop
while(num < len(list1)):

	# checking condition
	if list1[num] % 2 == 0: #if num % 2 != 0: for odd num in the list
		print(list1[num], end=" ")

	# increment num
	num += 1
       
#%%
#Method 3: Using list comprehension 
#------------------------------------

# Python program to print even Numbers in a List

list1 = [10, 21, 4, 45, 66, 93]

# using list comprehension
even_nos = [num for num in list1 if num % 2 == 0] #if num % 2 != 0: for odd num in the list

print("Even numbers in the list: ", even_nos)
    
#%%
#Method 4: Using lambda expressions 
#-------------------------------------

# Python program to print Even Numbers in a List

# list of numbers
list1 = [10, 21, 4, 45, 66, 93, 11]

# we can also print even no's using lambda exp.
even_nos = list(filter(lambda x: (x % 2 == 0), list1)) #x % 2 != 0: for odd num in the list

print("Even numbers in the list: ", even_nos)

#%%
#%%
#%%
#Python program to print all even/odd numbers in a range

start = int(input("Enter the start of range: "))
end = int(input("Enter the end of range: "))

#start, end = 4, 19

# iterating each number in list
for num in range(start, end + 1):	
	# checking condition
	if num % 2 == 0: #if num % 2 != 0: for odd numbers
		print(num, end = " ")


#%%
start = int(input("Enter the start of range: "))
end = int(input("Enter the end of range: "))
        
#create a list that contains only Even/Odd numbers in given range

#a[start:end:step]
#a[1::2] get every odd index, a[::2] get every even, a[2::2] get every even 
#starting at 2, a[2:4:2] get every even starting at 2 and ending at 4.


even_list = range(start, end + 1)[start%2::2] 
  
for num in even_list:
    print(num, end = " ")


#%%
xx = range(1,10,2)
print(list(xx))

#%%
list(range(10)[::3])

#%%
#%%
#%%        
#Python program to count Even or Odd numbers in a List
#---------------------------------------------------------

# list of numbers
list1 = [10, 21, 4, 45, 66, 93, 1]

even_count, odd_count = 0, 0

# iterating each number in list
for num in list1:	
	if num % 2 == 0:
		even_count += 1
	else:
		odd_count += 1
		
print("Even numbers in the list: ", even_count)
print("Odd numbers in the list: ", odd_count)


#%%
#Example 2: Using while loop
#------------------------------

list1 = [10, 21, 4, 45, 66, 93, 11]

even_count, odd_count = 0, 0

num = 0

# using while loop	
while(num < len(list1)):
	
	# checking condition
	if list1[num] % 2 == 0:
		even_count += 1
	else:
		odd_count += 1
	# increment num
	num += 1
	
print("Even numbers in the list: ", even_count)
print("Odd numbers in the list: ", odd_count)

#%%
#Example 3 : Using Python Lambda Expressions
#---------------------------------------------

# list of numbers
list1 = [10, 21, 4, 45, 66, 93, 11]

odd_count = len(list(filter(lambda x: (x%2 != 0) , list1)))

# we can also do len(list1) - odd_count
even_count = len(list(filter(lambda x: (x%2 == 0) , list1)))

print("Even numbers in the list: ", even_count)
print("Odd numbers in the list: ", odd_count)


#%%
#Example 4 : Using List Comprehension
#-------------------------------------------

# list of numbers
list1 = [10, 21, 4, 45, 66, 93, 11]

only_odd = [num for num in list1 if num % 2 == 1]
odd_count = len(only_odd)

print("Even numbers in the list: ", len(list1) - odd_count)
print("Odd numbers in the list: ", odd_count)

#%%
#%%
#%%
# Python program to count positive and negative numbers in a List

# list of numbers
list1 = [10, -21, 4, -45, 66, -93, 1]

pos_count, neg_count = 0, 0

# iterating each number in list
for num in list1:
	if num >= 0:
		pos_count += 1
	else:
		neg_count += 1
		
print("Positive numbers in the list: ", pos_count)
print("Negative numbers in the list: ", neg_count)

#%%
#Example #2: Using while loop
#---------------------------------
# list of numbers
list1 = [-10, -21, -4, -45, -66, 93, 11]

pos_count, neg_count = 0, 0
num = 0

# using while loop	
while(num < len(list1)):
	if list1[num] >= 0:
		pos_count += 1
	else:
		neg_count += 1
	num += 1
	
print("Positive numbers in the list: ", pos_count)
print("Negative numbers in the list: ", neg_count)

#%%
#Example #3 : Using Python Lambda Expressions
#-------------------------------------------------

# list of numbers
list1 = [10, -21, -4, 45, 66, 93, -11]

neg_count = len(list(filter(lambda x: (x < 0), list1)))

# we can also do len(list1) - neg_count
pos_count = len(list(filter(lambda x: (x >= 0), list1)))

print("Positive numbers in the list: ", pos_count)
print("Negative numbers in the list: ", neg_count)


#%%
#Example #4 : Using List Comprehension
#-------------------------------------------

# list of numbers
list1 = [-10, -21, -4, -45, -66, -93, 11]

only_pos = [num for num in list1 if num >= 1]
pos_count = len(only_pos)

print("Positive numbers in the list: ", pos_count)
print("Negative numbers in the list: ", len(list1) - pos_count)



#%%
#%%
#%%
#Python program to print positive/-ve numbers in a list
#------------------------------------------------------

# list of numbers
list1 = [11, -21, 0, 45, 66, -93]

# iterating each number in list
for num in list1:
    if num >=0: #if num < 0: for -ve numbers
        print(num, end = " ")

#%%
#Example #2: Using while loop
#-------------------------------
        
list1 = [-10, 21, -4, -45, -66, 93]
num = 0

while(num < len(list1)):
	if list1[num] >= 0: #if list1[num] < 0: for -ve numbers
		print(list1[num], end = " ")
	# increment num
	num += 1

#%%	
#Example #3: Using list comprehension
#----------------------------------------

# list of numbers
list1 = [-10, -21, -4, 45, -66, 93]

# using list comprehension
pos_nos = [num for num in list1 if num >= 0] #if num < 0 for -ve numbers

print("Positive numbers in the list: ", *pos_nos)
       
#%%
#Example #4: Using lambda expressions
#------------------------------------------

# list of numbers
list1 = [-10, 21, 4, -45, -66, 93, -11]

# we can also print positive no's using lambda exp.
pos_nos = list(filter(lambda x: (x >= 0), list1)) #(x < 0) for -ve numbers

print("Positive numbers in the list: ", *pos_nos)

#%%
#%%
#%%
#Remove multiple elements from a list in Python
#------------------------------------------------
# creating a list
list1 = [11, 5, 17, 18, 23, 50]

for ele in list1:
	if ele % 2 == 0:
		list1.remove(ele)

# printing modified list
print("New list after removing all even numbers: ", list1)




#%%
#Example #2: Using list comprehension
#--------------------------------------

# creating a list
list1 = [11, 5, 17, 18, 23, 50]

list1 = [ elem for elem in list1 if elem % 2 != 0]

print(*list1)




#%%
#Example #3: Remove adjacent elements using list slicing
#-------------------------------------------------------
# creating a list
list1 = [11, 5, 17, 18, 23, 50]

# removes elements from index 1 to 4 i.e. 5, 17, 18, 23 will be deleted
del list1[1:5]
print(*list1)




#%%
#Example #4: Using list comprehension (direct elements removal)
#----------------------------------------
# creating a list
list1 = [11, 5, 17, 18, 23, 50]

# items to be removed
unwanted_num = {11, 5}

list1 = [ele for ele in list1 if ele not in unwanted_num]

# printing modified list
print("New list after removing unwanted numbers: ", list1)




#%%
#Example #4: Using list comprehension (using index of elements we remove)

list1 = [11, 5, 17, 18, 23, 50]

# given index of elements removes 11, 18, 23
unwanted = [0, 3, 4]

list1 = [x for x in list1 if x not in map(list1.__getitem__, unwanted)]

#from operator import itemgetter
#list1 = [x for x in list1 if x not in list((itemgetter(*unwanted)(list1)))]

#import numpy as np
#list1 = [x for x in list1 if x not in list(np.array(list1)[unwanted])]

list1


#%%
#Example #5: When index of elements is known.
#-----------------------------------------------
list1 = [11, 5, 17, 18, 23, 50]

# given index of elements removes 11, 18, 23
unwanted = [0, 3, 4]

for ele in sorted(unwanted, reverse = True):
	del list1[ele]

# printing modified list
print (*list1)



#---------------------------------------------------------------------------------------
#%%
import numpy as np

ns = np.array(range(2,100))

primes = []
last_prime=2

while last_prime:
    primes.append(last_prime)
    ns = ns[ns%last_prime != 0]
    last_prime = ns[0] if len(ns) > 0 else None

print(primes[:100])

#%%
n=0
i=1
while n<100:
    i+=1
    count=1
    for j in range(2,i):
        if i%j==0:
            count=0
            break
    
    if count==1:
        print(i,end=' ')
        n+=1

#%%
#return vowel or consonent
def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'

grouped = df.groupby(get_letter_type, axis=1)

#%%






