import numpy as np
import pandas as pd

def original_X(X:np.ndarray, rows:np.ndarray, cols:np.ndarray, save_path:str):
    # save as npz
    save_data = {
            "X":X,
            "rows_vocabulary":rows,
            "cols_vocabulary":cols
    }
    np.savez_compressed(
        save_path
        + "/X_and_vocabulary"
        + ".npz",
        **save_data
    )
    # save as table
    A = np.hstack([rows.reshape(-1,1), X])
    B = np.vstack([np.hstack([["features/samples"], cols]).reshape(1,-1), A])   
    df = pd.DataFrame(B)
    headers = df.iloc[0]
    new_df  = pd.DataFrame(df.values[1:], columns=headers)
    new_df.to_csv(f'{save_path}/X_and_vocabulary_table.csv', index=False)

def X_tilda(W:np.ndarray, S:np.ndarray, H:np.ndarray, 
            bi:np.ndarray, bu:np.ndarray, global_mean:int, 
            rows:np.ndarray, cols:np.ndarray, save_path:str):
    
    if S is not None:
        X = np.add(np.add(W@S@H, bi).T, bu).T + global_mean
    else:
        X = np.add(np.add(W@H, bi).T, bu).T + global_mean

    save_data = {
            "X":X,
            "rows_vocabulary":rows,
            "cols_vocabulary":cols
    }
    np.savez_compressed(
        save_path
        + "/Xtilda_and_vocabulary"
        + ".npz",
        **save_data
    )
    # save as table
    A = np.hstack([rows.reshape(-1,1), X])
    B = np.vstack([np.hstack([["features/samples"], cols]).reshape(1,-1), A])   
    df = pd.DataFrame(B)
    headers = df.iloc[0]
    new_df  = pd.DataFrame(df.values[1:], columns=headers)
    new_df.to_csv(f'{save_path}/Xtilda_and_vocabulary_table.csv', index=False)
