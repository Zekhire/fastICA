# FastICA


#### Description
Project used to split N mixed audio files into N separated independent audio files. This project uses FastICA algorthm (not the most accurate one). Currently this project supports only .wav audio files.


#### Technologies
Project was created and tested with:
* Windows 10
* Python 3.6.5


#### Setup
- Run following block of commands in fastICA\ catalogue:
```
python -m virtualenv venv
cd venv
cd Scripts
activate
cd ..
cd ..
pip install -r requirements.txt
```
- Set all paths for audio files in "Editable parameters" section in fastica.py script
- Set all parameters in "Editable parameters" section in fastica.py script


#### Run
Go to fastICA\ and run command:
```
python fastica.py
```


#### References
This project is created based on following documents:
- https://eti.pg.edu.pl/documents/176593/26756916/STS.pdf (pages 43-44)
- A. Hyv¨arinen, E. Oja. “A fast fixed-point algorithm for independent component analysis”, Neural Computation, vol. 9, pp. 1483-1492, 1997.
