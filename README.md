### Installation

`git clone sso://user/eysenbach/LeaveNoTrace`

Install coach

`cd coach`
`git submodule init`
`git submodule update`
`./install.sh`
In the dialogue that appears, you only need to install the Coach and Gym
requirements.
`pip3 install gym tensorflow`
`touch coach/__init__.py`

### Usage

`source coach/coach_env/bin/activate`
`python3 demo.py`


### Debugging

I get the following error when installing coach:
```
Traceback (most recent call last):
  File "/usr/bin/pip3", line 9, in <module>
    from pip import main
ImportError: cannot import name 'main'
```
