# GPP Hack4Good
This is our team's submission for **Hack4Good hackathon** conducted at **[ KC College of Engineering, Thane ](https://www.kccemsr.edu.in)** on 28/04/2020.

### Our Team Members:
- Team Leader: Ankit Kumar BE COMPS 'A' 02 Email:kumarankit@kccemsr.edu.in
- Team Member: Pratik Chaudhari BE COMPS 'A' 06 Email:pratikchaudhari@kccemsr.edu.in
- Team Member: Kartik Bhat BE COMPS 'A' 04 Email: kartikbhat@kccemsr.edu.in

## Problem Statement:
The Garbage Profiling Problem looks at capturing images of the garbage at local garbage collection points and analyzing the same to create a rating for the community on the parameters of waste segregation. This analysis could be used to create feedback for the civic bodies to understand or identify communities where they need to take action to make sure this change is brought into action.


### Requirements:
- Python 3.7(64-bit)
- tensorflow==1.15.0
- Flask

### Installation
- For installation of TensorFlow Object Detection API see [ this ](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- Goto /models/research/ and use the following commands:
```
python setup.py build
python setup.py install
```
- Install the dependencies using the requirements.txt file using 
```
pip install -r requirements.txt
```
- Start the Flask Server using:
```
python app.py
```
### Note:
- The login id and password are Work in Progress. 
You can login using any combination of id and password.
- The live_inference.py fie can be used to get live predictions on video.
