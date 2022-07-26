# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:33:39 2021

@author: rvamsikrishna
"""

#A function is a group of related statements that performs a specific task.

#Functions help break our program into smaller and modular chunks. 
#As our program grows larger and larger, functions make it more
# organized and manageable.

#Furthermore, it avoids repetition and makes the code reusable.

#Types of Functions
#-------------------------
#Built-in functions - Functions that are built into Python.
#User-defined functions - Functions defined by the users themselves.

#%%
#Syntax of Function
def function_name(parameters):
	"""docstring"""
	statement(s)
    
# Keyword def that marks the start of the function header.
# A function name to uniquely identify the function.    
# Parameters (arguments) through which we pass values to a function. They are optional.
# A colon (:) to mark the end of the function header.
# Optional documentation string (docstring) to describe what the function does.
#One or more valid python statements that make up the function body. 
#Statements must have the same indentation level (usually 4 spaces).
#An optional return statement to return a value from the function.    
        
    
#%%
#Example of a function
def greet(name):
    """
    This function greets to
    the person passed in as
    a parameter
    """
    print("Hello, " + name + ". Good morning!")
    

#%%
#How to call a function in python?
#Once we have defined a function, we can call it from another function, 
#program, or even the Python prompt. To call a function we simply type 
#the function name with appropriate parameters.
greet('Paul')    


#%%
#Docstrings
#The first string after the function header is called the docstring 
#and is short for documentation string. It is briefly used to explain 
#what a function does.


#This string is available to us as the __doc__ attribute of the function.
print(greet.__doc__)


#%%
#The return statement
#The return statement is used to exit a function and go back to 
#the place from where it was called.

#This statement can contain an expression that gets evaluated and the 
#value is returned. 
#If there is no expression in the statement or the return statement itself 
#is not present inside a function, then the function will return the None object.

print(greet("May"))

#Here, None is the returned value since greet() directly prints the
# name and no return statement is used.

#%%
#Example of return
def absolute_value(num):
    """This function returns the absolute
    value of the entered number"""

    if num >= 0:
        return num
    else:
        return -num


print(absolute_value(2))

print(absolute_value(-4))

#%%
#Scope and Lifetime of variables
#-----------------------------
#Scope of a variable is the portion of a program where the variable 
#is recognized. Parameters and variables defined inside a function are
# not visible from outside the function. Hence, they have a local scope.

#The lifetime of a variable is the period throughout which the variable 
#exits in the memory. The lifetime of variables inside a function is as 
#long as the function executes.

#They are destroyed once we return from the function. Hence, a function
# does not remember the value of a variable from its previous calls.

#Here is an example to illustrate the scope of a variable inside a function.

def my_func():
	x = 10
	print("Value inside function:",x)


x = 20
my_func()
print("Value outside function:",x)


#Here, we can see that the value of x is 20 initially. Even though the 
#function my_func() changed the value of x to 10, it did not affect 
#the value outside the function.

#This is because the variable x inside the function is different
# (local to the function) from the one outside. Although they have 
#the same names, they are two different variables with different scopes.

#On the other hand, variables outside of the function are visible
# from inside. They have a global scope.

#We can read these values from inside the function but cannot change 
#(write) them. In order to modify the value of variables outside the 
#function, they must be declared as global variables using the
# keyword global.



#%%
#%%
#%%
#Python Function Arguments
#==========================

#In Python, you can define a function that takes variable number of arguments.

#Here you will learn to define such functions using default,
# keyword and arbitrary arguments.

#Arguments
#-----------
#In the user-defined function topic, we learned about defining a function 
#and calling it. Otherwise, the function call will result in an error. 
#Here is an example.
def greet(name, msg):
    """This function greets to
    the person with the provided message"""
    print("Hello", name + ', ' + msg)

greet("Monica", "Good morning!")

greet("Monica")    # only one argument (returns positional argument Error)

greet()    # no arguments (Returns positional argument Error)


#%%
#Variable Function Arguments
#===========================
#Up until now, functions had a fixed number of arguments. In Python, 
#there are other ways to define a function that can take variable number of arguments.

#Three different forms of this type are described below.

#Python Default Arguments
#=========================
#Function arguments can have default values in Python.
#We can provide a default value to an argument by using the assignment operator (=). 

def greet(name, msg="Good morning!"):
    """
    This function greets to the person with the provided message.

    If the message is not provided,it defaults to "Good morning!"
    """

    print("Hello", name + ', ' + msg)


greet("Kate")
greet("Bruce", "How do you do?")

#Any number of arguments in a function can have a default value
#default argument must follow non-default argument otherwise it throw an error

# def greet(msg = "Good morning!", name): (trrow an error)

#%%
#Python Keyword Arguments
#===========================
#When we call a function with some values, these values get assigned
# to the arguments according to their position.

#For example, in the above function greet(),
# when we called it as greet("Bruce", "How do you do?"), 
#the value "Bruce" gets assigned to the argument name and 
#similarly "How do you do?" to msg.

#Python allows functions to be called using keyword arguments. 
#When we call functions in this way, the order (position) of the 
#arguments can be changed

# 2 keyword arguments
greet(name = "Bruce",msg = "How do you do?")

# 2 keyword arguments (out of order)
greet(msg = "How do you do?",name = "Bruce") 

# 1 positional, 1 keyword argument
greet("Bruce", msg = "How do you do?")  

#we must keep in mind that keyword arguments must follow positional arguments.

#Having a positional argument after keyword arguments will result in errors. 

greet(name="Bruce","How do you do?")

#Will result in an error:SyntaxError: non-keyword arg after keyword arg

#%%
#Python Arbitrary Arguments
#============================
#Sometimes, we do not know in advance the number of arguments that will 
#be passed into a function. Python allows us to handle this kind of 
#situation through function calls with an arbitrary number of arguments.

#In the function definition, we use an asterisk (*) before the parameter 
#name to denote this kind of argument. Here is an example.

def greet(*names):
    """This function greets all
    the person in the names tuple."""

    # names is a tuple with arguments
    for name in names:
        print("Hello", name)


greet("Monica", "Luke", "Steve", "John")

# Here, we have called the function with multiple arguments.
# These arguments get wrapped up into a tuple before being passed 
# into the function. Inside the function, we use a for loop to 
# retrieve all the arguments back.


#%%
#%%
#%%
#Python Recursion
#===================
#recursive function (a function that calls itself).

#Ex : two parallel mirrors facing each other. Any object in between
# them would be reflected recursively.

#Python Recursive Function
#=========================
#In Python, we know that a function can call other functions. 
#It is even possible for the function to call itself. These types 
#of construct are termed as recursive functions.

#Factorial of a number is the product of all the integers from 1 to that 
#number. For example, the factorial of 6 (denoted as 6!) is 1*2*3*4*5*6 = 720.

def factorial(x):
    """This is a recursive function
    to find the factorial of an integer"""

    if x == 1:
        return 1
    else:
        return (x * factorial(x-1))


num = 3
print("The factorial of", num, "is", factorial(num))

#When we call this function with a positive integer, it will 
#recursively call itself by decreasing the number.

#*******
#Every recursive function must have a base condition that stops 
#the recursion or else the function calls itself infinitely.

#The Python interpreter limits the depths of recursion to help 
#avoid infinite recursions, resulting in stack overflows.

#By default, the maximum depth of recursion is 1000. If the limit is
#crossed, it results in RecursionError. Let's look at one such condition.


#Advantages of Recursion
#-------------------------
#Recursive functions make the code look clean and elegant.
#A complex task can be broken down into simpler sub-problems using recursion.
#Sequence generation is easier with recursion than using some nested iteration.

#Disadvantages of Recursion
#---------------------------
# Sometimes the logic behind recursion is hard to follow through.
# Recursive calls are expensive (inefficient) as they take up a
# lot of memory and time.
# Recursive functions are hard to debug.


#%%
#%%
#%%
#Python Anonymous/Lambda Function
#=================================

#What are lambda functions in Python?

#In Python, an anonymous function is a function that is defined without a name.

#While normal functions are defined using the def keyword in Python, 
#anonymous functions are defined using the lambda keyword.

#Hence, anonymous functions are also called lambda functions.

#Syntax of Lambda Function in python
lambda arguments: expression

#Lambda functions can have any number of arguments but only one expression. 
#The expression is evaluated and returned. 
#Lambda functions can be used wherever function objects are required.

# Program to show the use of lambda functions
double = lambda x: x * 2

print(double(5))

#In the above program, lambda x: x * 2 is the lambda function. 
#Here x is the argument and x * 2 is the expression that gets 
#evaluated and returned.

#This function has no name. It returns a function object which is assigned 
#to the identifier double. We can now call it as a normal function.

double = lambda x: x * 2

#is nearly the same as:

def double(x):
   return x * 2


#Use of Lambda Function in python
#-----------------------------------
#We use lambda functions when we require a nameless function 
#for a short period of time.
   
#********
#In Python, we generally use it as an argument to a higher-order
#function (a function that takes in other functions as arguments).
#Lambda functions are used along with built-in functions 
#like filter(), map() etc.


#%%   
#Example use with filter()
#--------------------------
#The filter() function in Python takes in a function and a list as arguments.

#The function is called with all the items in the list and a new list is
#returned which contains items for which the function evaluates to True.
   
# Program to filter out only the even items from a list
my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(filter(lambda x: (x%2 == 0) , my_list))

print(new_list)

#%%
#Example use with map()
#---------------------------
#The map() function in Python takes in a function and a list.

#The function is called with all the items in the list and a new list 
#is returned which contains items returned by that function for each item.


# Program to double each item in a list using map()
my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(map(lambda x: x * 2 , my_list))

print(new_list)


#%%
#%%
#%%
#Python Global, Local and Nonlocal variables
#=============================================

#Global Variables
#-----------------
#In Python, a variable declared outside of the function or in
# global scope is known as a global variable. This means that a 
# global variable can be accessed inside or outside of the function.

#Example 1: Create a Global Variable
x = "global"

def foo():
    print("x inside:", x)


foo()
print("x outside:", x)

#In the above code, we created x as a global variable and defined 
#a foo() to print the global variable x. Finally, we call the foo() 
#which will print the value of x.

#%%
#What if you want to change the value of x inside a function?

x = "global"

def foo():
    x = x * 2
    print(x)

foo()

#Output : UnboundLocalError: local variable 'x' referenced before assignment

#The output shows an error because Python treats x as a local variable
# and x is also not defined inside foo().

#To make this work, we use the global keyword. 
#Visit Python Global Keyword to learn more.
#%%
x = " global"

def foo():
    global x
    x = x * 2
    print(x)

foo()


#%%
#Local Variables
#-------------------
#A variable declared inside the function's body or in the local scope
# is known as a local variable.

#%%
#Example 2: Accessing local variable outside the scope
def foo():
    y = "local"


foo()
print(y)

#Output : NameError: name 'y' is not defined

#The output shows an error because we are trying to access a 
#local variable y in a global scope whereas the local variable 
#only works inside foo() or local scope.
#%%
#Example 3: Create a Local Variable
#------------------------------------
def foo():
    y = "local"
    print(y)

foo()

#%%
#Example 4: Using Global and Local variables in the same code
x = "global "

def foo():
    global x
    y = "local"
    x = x * 2
    print(x)
    print(y)

foo()

print(x)

#In the above code, we declare x as a global and y as a local variable
# in the foo(). Then, we use multiplication operator * to modify the
# global variable x and we print both x and y.


#After calling the foo(), the value of x becomes global global because 
#we used the x * 2 to print two times global. After that, we print the 
#value of local variable y i.e local.

#%%
#Example 5: Global variable and Local variable with same name
x = 5

def foo():
    x = 10
    print("local x:", x)


foo()
print("global x:", x)
    

#%%
#Nonlocal Variables
#========================

#Nonlocal variables are used in nested functions whose local scope is 
#not defined. This means that the variable can be neither in the 
#local nor the global scope.

#We use nonlocal keywords to create nonlocal variables.

#Example 6: Create a nonlocal variable
def outer():
    x = "local"

    def inner():
        nonlocal x
        x = "nonlocal"
        print("inner:", x)

    inner()
    print("outer:", x)


outer()

#Output
#inner: nonlocal
#outer: nonlocal    

#In the above code, there is a nested inner() function.
# We use nonlocal keywords to create a nonlocal variable. 
#The inner() function is defined in the scope of another function outer().


#Note : If we change the value of a nonlocal variable, 
#the changes appear in the local variable.


#%%
#%%
#%%
#Python Global Keyword 
#=========================

#What is the global keyword
#In Python, global keyword allows you to modify the variable 
#outside of the current scope. It is used to create a global variable 
#and make changes to the variable in a local context.

#Rules of global Keyword
#--------------------------
#When we create a variable inside a function, it is local by default.

#When we define a variable outside of a function, it is global
# by default. You don't have to use global keyword.

#We use global keyword to read and write a global variable inside a function.

#Use of global keyword outside a function has no effect.

#%%
#Use of global Keyword
#----------------------
#Example 1: Accessing global Variable From Inside a Function
c = 1 # global variable

def add():
    print(c)

add()

#Outputs : 1

#However, we may have some scenarios where we need to modify 
#the global variable from inside a function.

#%%
#Example 2: Modifying Global Variable From Inside the Function
c = 1 # global variable
    
def add():
    c = c + 2 # increment c by 2
    print(c)

add()

#Outputs : UnboundLocalError: local variable 'c' referenced before assignment

#This is because we can only access the global variable but 
#cannot modify it from inside the function.

#The solution for this is to use the global keyword.

#%%
#Example 3: Changing Global Variable From Inside a Function using global
c = 0 # global variable

def add():
    global c
    c = c + 2 # increment by 2
    print("Inside add():", c)

add()
print("In main:", c)

#As we can see, change also occurred on the global variable
# outside the function, c = 2.

#%%
#Global Variables Across Python Modules
#==========================================

#In Python, we create a single module config.py to hold 
#global variables and share information across Python modules 
#within the same program.

#Here is how we can share global variables across the python modules.

#%%
#Example 4: Share a global Variable Across Python Modules

#Create a config.py file, to store global variables
a = 0
b = "empty"

#Create a update.py file, to change global variables
import config

config.a = 10
config.b = "alphabet"

#Create a main.py file, to test changes in value
import config
import update

print(config.a)
print(config.b)


#When we run the main.py file, the output will be ; 10,alphabet

#%%
#Global in Nested Functions
#-----------------------------
# Here is how you can use a global variable in nested function.

#Example 5: Using a Global Variable in Nested Function
def foo():
    x = 20

    def bar():
        global x
        x = 25
    
    print("Before calling bar: ", x)
    print("Calling bar now")
    bar()
    print("After calling bar: ", x)

foo()
print("x in main: ", x)    

#In the above program, we declared a global variable inside the 
#nested function bar(). Inside foo() function, x has no effect 
#of the global keyword.

#Before and after calling bar(), the variable x takes the value of
# local variable i.e x = 20. Outside of the foo() function, the 
#variable x will take value defined in the bar() function i.e x = 25.
# This is because we have used global keyword in x to create global 
#variable inside the bar() function (local scope).

#If we make any changes inside the bar() function, the changes 
#appear outside the local scope, i.e. foo().


#%%
#%%
#%%
#Python Modules
#===============
#In this article, you will learn to create and import custom modules 
#in Python. Also, you will find different techniques to import and
# use custom and built-in modules in Python.

#What are modules in Python?
#---------------------------

#Modules refer to a file containing Python statements and definitions.

#A file containing Python code, for example: example.py, is called a 
#module, and its module name would be example.

#We use modules to break down large programs into small manageable and 
#organized files. Furthermore, modules provide reusability of code.

#We can define our most used functions in a module and import it, 
#instead of copying their definitions into different programs.

#Let us create a module. Type the following and save it as example.py.

# Python Module example

def add(a, b):
   """This program adds two
   numbers and return the result"""

   result = a + b
   return result

#%%
#How to import modules in Python?
#-----------------------------------
#We can import the definitions inside a module to another module or 
#the interactive interpreter in Python.
   
# To import our previously defined module example,
# we type the following in the Python prompt.
 
import example

#Using the module name we can access the function using the dot . operator. 
#For example:
example.add(4,5.5)

#Python has tons of standard modules. You can check out the full 
#list of Python standard modules and their use cases. 
#These files are in the Lib directory inside the location where 
#you installed Python.
   
#Standard modules can be imported the same way as we import our
# user-defined modules.

#There are various ways to import modules. They are listed below..

#Python import statement
#--------------------------
#We can import a module using the import statement and access the 
#definitions inside it using the dot operator as described above. 
#Here is an example.

#%%
# import statement example to import standard module math

import math
print("The value of pi is", math.pi)

#%%
#Import with renaming
#----------------------
# import module by renaming it

import math as m
print("The value of pi is", m.pi)

#%%
#Python from...import statement
#-------------------------------
#We can import specific names from a module without importing the 
#module as a whole. Here is an example.
# import only pi from math module

from math import pi
print("The value of pi is", pi)

#We can also import multiple attributes as follows:
from math import pi, e

#%%
#Import all names
#------------------
from math import *

#Here, we have imported all the definitions from the math module.
# This includes all names visible in our scope except those
# beginning with an underscore(private definitions).

#Importing everything with the asterisk (*) symbol is not a good programming
# practice. This can lead to duplicate definitions for an identifier. 
#It also hampers the readability of our code.

#%%
#Python Module Search Path
#==============================
#While importing a module, Python looks at several places.
# Interpreter first looks for a built-in module. 
#Then(if built-in module not found), Python looks into a list 
#of directories defined in sys.path. The search is in this order.

#The current directory.
#PYTHONPATH (an environment variable with a list of directories).
#The installation-dependent default directory.
import sys
sys.path

#We can add and modify this list to add our own path.

#%%
#Reloading a module
#=====================

#The Python interpreter imports a module only once during a session. 
#This makes things more efficient. Here is an example to show how this works.

#Now if our module changed (additional functions created inside the module)
#during the course of the program, we would have to reload it.
#One way to do this is to restart the interpreter.
#But this does not help much.

#Python provides a more efficient way of doing this. We can use
# the reload() function inside the "imp" module to reload a module. 
#We can do it in the following ways:

import imp
import my_module
import my_module
imp.reload(my_module)

#%%
#The dir() built-in function
#==============================

#We can use the dir() function to find out names that are 
#defined inside a module.

#For example, we have defined a function add() in the
# module example that we had in the beginning.

dir(example)

#Output::
'''
['__builtins__',
'__cached__',
'__doc__',
'__file__',
'__initializing__',
'__loader__',
'__name__',
'__package__',
'add']
'''

#Here, we can see a sorted list of names (along with add). 
#All other names that begin with an underscore are default
# Python attributes associated with the module (not user-defined).

#For example, the __name__ attribute contains the name of the module.

import example
example.__name__

#All the names defined in our current namespace can be found out 
#using the dir() function without any arguments.

a=1
b="Hello"
import math
dir()

#%%
#%%
#%%
#Python Package
#===================

#divide your code base into clean, efficient modules using Python
# packages. Also, you'll learn to import and use your own or 
#third party packages in a Python program.

#What are packages?
#-----------------

#We don't usually store all of our files on our computer in 
#the same location. We use a well-organized hierarchy of 
#directories for easier access.

#Similar files are kept in the same directory, for example, we may keep
# all the songs in the "music" directory. Analogous to this, Python has
# packages for directories and modules for files.

#As our application program grows larger in size with a lot of modules,
# we place similar modules in one package and different modules in 
#different packages. This makes a project (program) easy to manage 
#and conceptually clear.

#Similarly, as a directory can contain subdirectories and files,
# a Python package can have sub-packages and modules.

#A directory must contain a file named __init__.py in order for 
#Python to consider it as a package. This file can be left empty 
#but we generally place the initialization code for that package 
#in this file.

#Importing module from a package
#------------------------------
#We can import modules from packages using the dot (.) operator.
#For example, if we want to import the start module in the above example, 
#it can be done as follows:
import Game.Level.start

#Now, if this module contains a function named select_difficulty(), 
#we must use the full name to reference it.
Game.Level.start.select_difficulty(2)


#If this construct seems lengthy, we can import the module without 
#the package prefix as follows:
from Game.Level import start


#We can now call the function simply as follows:
start.select_difficulty(2)


#Another way of importing just the required function (or class or variable)
# from a module within a package would be as follows:
from Game.Level.start import select_difficulty


#Now we can directly call this function.
select_difficulty(2)

#Although easier, this method is not recommended. Using the full 
#namespace avoids confusion and prevents two same identifier names
#from colliding.

#While importing packages, Python looks in the list of directories 
#defined in sys.path, similar as for module search path.























    
    