import numpy as np
import pandas as pd
import os

#lol thanks chat

#file = r'C:\Code\brainrender\to_analyze\swc\N022-674185-HD.swc'
def load_swc(file_path):
    """Loads an SWC file into a pandas DataFrame."""
    df = pd.read_csv(file_path, comment='#', sep=' ',
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
    return df

def save_swc(df, file_path):
    """Saves a pandas DataFrame to an SWC file."""
    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{int(row['id'])} {int(row['type'])} {row['x']} {row['y']} {row['z']} {row['radius']} {int(row['parent'])}\n")

# Example usage
if __name__ == '__main__':
    # Load the SWC file
    folder_path = r'C:\Data\reconstructions\medulla_IRN_PRN_PGRN\medulla_IRN_PRN_PGRN\swc\down'
    for filename in os.listdir(folder_path):
        file = os.path.join(folder_path, filename)
        data = load_swc(file)

        # Modify the data (optional)
        # For example, scale the coordinates by a factor of 2
        # data[['x', 'y', 'z']] = data[['x', 'y', 'z']] * 2
        data[['x','y','z']] = data [['z','y','x']]
        # Save the modified data back to a new SWC file
        #new_filename = 'modified_'+filename
        savefolder=r"C:\Data\reconstructions\medulla_IRN_PRN_PGRN\medulla_IRN_PRN_PGRN\swccoordswapped"
        output_file = savefolder+filename
        save_swc(data, output_file)
        print(f"SWC file saved to {output_file}")