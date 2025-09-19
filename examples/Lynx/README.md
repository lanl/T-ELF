## Usage:
1. ```cd``` into the root directory of this repository

    ```cd TELF```
2. Create a ```projects``` directory.

    ```mkdir projects```
3. Put your post-processing folder for each project under this directory.

    ```cp -r /path/to/project1 TELF/projects```
4. On terminal start the server

    ```streamlit run TELF/applications/Lynx/frontend/main.py```
5. **Optional** if running on a remote server, forward the ports by running the following on the local terminal 

    ```ssh USER@HOST -L 8501:localhost:8501```
6. Have fun!

