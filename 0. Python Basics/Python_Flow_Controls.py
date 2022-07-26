# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 09:23:15 2019

@author: rvamsikrishna
"""

# Python if...else Statement
#-------------------------------

#The if…elif…else statement is used in Python for decision making.

#Python if Statement Syntax

#if test expression:
#    statement(s)

#Here, the program evaluates the test expression and will execute statement(s) 
#only if the text expression is True.

#If the text expression is False, the statement(s) is not executed.

#In Python, the body of the if statement is indicated by the indentation. 
#Body starts with an indentation and the first unindented line marks the end.

#Python interprets non-zero values as True. None and 0 are interpreted as False.

# If the number is positive, we print an appropriate message

num = 3
if num > 0:
    print(num, "is a positive number.")
print("This is always printed.")

num = -1
if num > 0:
    print(num, "is a positive number.")
print("This is also always printed.")


#%%

#Python if...else Statement
#---------------------------
#if test expression:
#    Body of if
#else:
#    Body of else

#The if..else statement evaluates test expression and will execute body of 
#if only when test condition is True.

#If the condition is False, body of else is executed. Indentation is used 
#to separate the blocks.

num = 3

# Try these two variations as well. 
# num = -5
# num = 0

if num >= 0:
    print("Positive or Zero")
else:
    print("Negative number")


#%%

#Python if...elif...else Statement
#------------------------------------

#if test expression:
#    Body of if
#elif test expression:
#    Body of elif
#else: 
#    Body of else    

#The elif is short for else if. It allows us to check for multiple expressions.
#If the condition for if is False, it checks the condition of the next elif block and so on.
#If all the conditions are False, body of else is executed.
#Only one block among the several if...elif...else blocks is executed according to the condition.
#The if block can have only one else block. But it can have multiple elif blocks.

num = 3.4

# Try these two variations as well:
# num = 0
# num = -4.5

if num > 0:
    print("Positive number")
elif num == 0:
    print("Zero")
else:
    print("Negative number")

#%%

#Python Nested if statements
#-----------------------------
#We can have a if...elif...else statement inside another if...elif...else statement. 
#This is called nesting in computer programming.
    
#Any number of these statements can be nested inside one another. 
#Indentation is the only way to figure out the level of nesting.
    
num = float(input("Enter a number: "))
if num >= 0:
    if num == 0:
        print("Zero")
    else:
        print("Positive number")
else:
    print("Negative number")
    
#%%    
#%%
#%%
    
#Python for Loop
#------------------    

#The for loop in Python is used to iterate over a sequence (list, tuple, string) or 
#other iterable objects. Iterating over a sequence is called traversal.    

#for val in sequence:
#	Body of for

#Here, val is the variable that takes the value of the item inside the
# sequence on each iteration.
    
#Loop continues until we reach the last item in the sequence. 
#The body of for loop is separated from the rest of the code using indentation.
    
#%%    
# Program to find the sum of all numbers stored in a list

# List of numbers
numbers = [6, 5, 3, 8, 4, 2, 5, 4, 11]

# variable to store the sum
sum = 0

# iterate over the list
for val in numbers:
	sum = sum+val

# Output: The sum is 48
print("The sum is", sum)

#%%
#The range() function
#-----------------------

#We can generate a sequence of numbers using range() function. range(10) will 
#generate numbers from 0 to 9 (10 numbers).

#We can also define the start, stop and step size as range(start,stop,step size). 
#step size defaults to 1 if not provided.

#This function does not store all the values in memory, it would be inefficient.
# So it remembers the start, stop, step size and generates the next number on the go.

#To force this function to output all the items, we can use the function list().

print(range(10))
print(list(range(10)))
print(list(range(2, 8)))
print(list(range(2, 20, 3)))

#We can use the range() function in for loops to iterate through a sequence of numbers. 
#It can be combined with the len() function to iterate though a sequence using indexing.

# Program to iterate through a list using indexing
genre = ['pop', 'rock', 'jazz']

# iterate over the list using index
for i in range(len(genre)):
	print("I like", genre[i])



#%%
#for loop with else
#=====================    
#A for loop can have an optional else block as well. The else part is executed
# if the items in the sequence used in for loop exhausts.
#break statement can be used to stop a for loop. In such case, the else part is ignored.
#Hence, a for loop's else part runs if no break occurs.
    
digits = [0, 1, 5]

for i in digits:
    print(i)
else:
    print("No items left.")
    
    
#%%    
#%%
#%%
#Python while Loop
#===================

#https://snakify.org/en/lessons/while_loop/
    
#while loop repeats the sequence of actions many times until some condition evaluates to False. 

#The condition is given before the loop body and is checked before each execution of the loop body. 

#Typically, the while loop is used when it is impossible to determine the exact number 
#of loop iterations in advance.
    
#Python firstly checks the condition. If it is False, then the loop is terminated and control
# is passed to the next statement after the while loop body
    
#If the condition is True, then the loop body is executed, and then the condition is checked again.

#This continues while the condition is True. Once the condition becomes False, the loop 
#terminates and control is passed to the next statement after the loop.
 
i = 1
while i <= 10:
    print(i ** 2)
    i += 1  

#%%
n = int(input())
length = 0
while n > 0:
    n //= 10  # this is equivalent to n = n // 10
    length += 1
print(length)  # 4"    
#%%
#One can write an else: statement after a loop body which is executed once after the end of the loop:

i = 1
while i <= 10:
    print(i)
    i += 1
else:
    print('Loop ended, i =', i)
    
#%%    

#The while loop in Python is used to iterate over a block of code as long as 
#the test expression (condition) is true.
    
#We generally use this loop when we don't know beforehand, the number of times to iterate.
    
#while test_expression:
#    Body of while
    
#In while loop, test expression is checked first. The body of the loop is entered only 
#if the test_expression evaluates to True. After one iteration, the test expression is
# checked again. This process continues until the test_expression evaluates to False.
    
#In Python, the body of the while loop is determined through indentation.
#Body starts with indentation and the first unindented line marks the end.
#Python interprets any non-zero value as True. None and 0 are interpreted as False.


# Program to add natural numbers upto  sum = 1+2+3+...+n
    
# To take input from the user,
# n = int(input("Enter n: "))

n = 10

# initialize sum and counter
sum = 0
i = 1

while i <= n:
    sum = sum + i
    i = i+1    # update counter

# print the sum
print("The sum is", sum)

#%%
#while loop with else
#-----------------------

#The else part is executed if the condition in the while loop evaluates to False.

#The while loop can be terminated with a break statement.
# In such case, the else part is ignored.
# Hence, a while loop's else part runs if no break occurs and the condition is false

counter = 0

while counter < 3:
    print("Inside loop")
    counter = counter + 1
else:
    print("Inside else")
    
    
#%%
#%%
#%%
#Python break and continue
#========================== 

#In Python, break and continue statements can alter the flow of a normal loop.
    
#Loops iterate over a block of code until test expression is false, but sometimes 
#we wish to terminate the current iteration or even the whole loop without checking 
#test expression.
    
#Python break statement
#----------------------------    

#The break statement terminates the loop containing it. Control of the
#program flows to the statement immediately after the body of the loop.
    
#If break statement is inside a nested loop (loop inside another loop), 
#break will terminate the innermost loop.
    
# Use of break statement inside loop

for val in "string":
    if val == "i":
        break
    print(val)

print("The end")

#%%
#Python continue statement
#-----------------------------

#The continue statement is used to skip the rest of the code inside 
#a loop for the current iteration only. Loop does not terminate but 
#continues on with the next iteration.

# Program to show the use of continue statement inside loops

for val in "string":
    if val == "i":
        continue
    print(val)

print("The end")

#%%
#Python pass statement
#-----------------------------

#In Python programming, pass is a null statement. The difference between a
# comment and pass statement in Python is that, while the interpreter ignores
# a comment entirely, pass is not ignored.

#However, nothing happens when pass is executed. It results into no operation (NOP).

#Suppose we have a loop or a function that is not implemented yet, but we want
# to implement it in the future. They cannot have an empty body.
# The interpreter would complain. So, we use the pass statement to 
#construct a body that does nothing

sequence = {'p', 'a', 's', 's'}
for val in sequence:
    pass


#We can do the same thing in an empty function or class as well.
def function(args):
    pass

#-----------------
class example:
    pass
    

#%%
#%%
#%%
#Python Looping Techniques
#===========================    

#Python programming offers two kinds of loop, the for loop and the while loop.
#Using these loops along with loop control statements like break and continue, 
#we can create various forms of loop.
    
#The infinite loop
#-------------------

#We can create an infinite loop using while statement. 
#If the condition of while loop is always True, we get an infinite loop.
    
# An example of infinite loop
# press Ctrl + c to exit from the loop
while True:
    num = int(input("Enter an integer: "))
    print("The double of",num,"is",2 * num)


#Loop with condition at the top
#-------------------------------------
# Program to illustrate a loop with condition at the top

# Try different numbers
n = 10

# Uncomment to get user input
#n = int(input("Enter n: "))

# initialize sum and counter
sum = 0
i = 1

while i <= n:
   sum = sum + i
   i = i+1    # update counter

# print the sum
print("The sum is",sum)


#Loop with condition in the middle
#------------------------------------------
# Program to illustrate a loop with condition in the middle. 
# Take input from the user untill a vowel is entered
vowels = "aeiouAEIOU"
# infinite loop
while True:
   v = input("Enter a vowel: ")
   # condition in the middle
   if v in vowels:
      break
   print("That is not a vowel. Try again!")

print("Thank you!")    


#Loop with condition at the bottom
#---------------------------------------
# Python program to illustrate a loop with condition at the bottom
# Roll a dice untill user chooses to exit
# import random module
import random
while True:
   input("Press enter to roll the dice")
   # get a number between 1 to 6
   num = random.randint(1,6)
   print("You got",num)
   option = input("Roll again?(y/n) ")
   # condition
   if option == 'n':
       break
 
#%%       
#%%
#%%
#%%
"""
Created on Sun Jul 21 01:08:06 2019

@author: rvamsikrishna
"""
#What is pass statement in Python?
#pass is a null statement. The difference between a comment and pass statement
#in Python is that, while the interpreter ignores a comment entirely, pass is not ignored.

#It is used when the statement is required syntactically, but you do not want that 
#code to execute. The pass statement is the null operation; nothing happens when it runs.

#The pass statement is also useful in scenarios where your code will eventually go 
#but has not been entirely written yet

#Suppose we have a loop or a function that is not implemented yet, but we want 
#to implement it in the future.

#They cannot have an empty body. The interpreter would complain.

#So, we use the pass statement to construct a body that does nothing.

#nothing happens when pass is executed. It results into no operation (NOP)


#---------------------------------
#Python Language has the syntactical requirement that code blocks like
# if, except, def, class, etc. cannot be empty. Empty code blocks are however 
#useful in the variety of different contexts in the program.

#Therefore, if nothing is supposed to happen in the code, the pass statement 
#is needed for that block not to produce the IndentationError.

# Alternatively, any statement (including just a term to be evaluated, 
#like the Ellipsis literal … or a string, most often a docstring) can be used,
#but the pass statement makes clear that indeed nothing is supposed to happen in
# that block of code, and does not need to be actually run and (at least temporarily) 
#stored in the memory.



#%%

# pass is just a placeholder for
# functionality to be added later.
sequence = {'p', 'a', 's', 's'}
for val in sequence:
    pass

#%%
#We can do the same thing in an empty function or class as well.
def function(args):
    pass    


#%%
class example:
    pass

#The most use case of the pass statement is the scenario where you are designing a
# new class with some methods that you do not want to implement, yet.
    

#%%
for letter in 'Python': 
   if letter == 'h':
      pass
      print ('This is pass block')
   print('Current Letter :', letter)

print ("Good bye!")

#%%

################################################################################################
#%%
#======================================================================================
#Iterator Function 
#=========================

A = [1,2,3,4]
B = iter(A) # this builds an iterator object

#%%
print(type(A))

print(type(B))

#%%
A = [1,2,3,4]
B = iter(A) # this builds an iterator object
#%%
print (next(B)) #prints next available element in iterator

#%%
A = [1,2,3,4]
B = iter(A) # this builds an iterator object

#Iterator object can be traversed using regular for statement
for x in B:
   print (x, end=" ")
   
#%%
import sys
#%%
A = [1,2,3,4]
B = iter(A) # this builds an iterator object

while True:
   try:
      print(next(B))
   except StopIteration:
      sys.exit() #you have to import sys module for this
      
#%%
#======================================================================================
#generator  function
#=========================
import sys
def fibonacci(n): #generator function
   a, b, counter = 0, 1, 0
   while True:
      if (counter > n): 
         return
      yield a
      a, b = b, a + b
      counter += 1
      
#%%      
f = fibonacci(5) #f is iterator object

#%%
type(f)

#%%
print (next(f), end=" ")

#%%

f = fibonacci(7) #f is iterator object

while True:
   try:
      print (next(f), end=" ")
   except StopIteration:
      sys.exit()

#%%

       



























































