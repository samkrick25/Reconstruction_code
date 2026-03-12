import pandas as pd

def swap_xz_coordinates(input_file, output_file):
    """
    Swap x and z coordinates in an SWC file.
    
    SWC format: n T x y z R P
    where n=sample number, T=structure identifier, x,y,z=coordinates,
    R=radius, P=parent sample number
    """
    # Read the file, preserving comment lines
    comments = []
    data_lines = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                comments.append(line)
            elif line.strip():  # non-empty, non-comment line
                data_lines.append(line)
    
    # Parse data into DataFrame
    # SWC columns: n, T, x, y, z, R, P
    data = []
    for line in data_lines:
        values = line.split()
        data.append(values)
    
    df = pd.DataFrame(data, columns=['n', 'T', 'x', 'y', 'z', 'R', 'P'])
    
    # Swap x and z columns
    df['x'], df['z'] = df['z'].copy(), df['x'].copy()
    
    # Write to output file
    with open(output_file, 'w') as f:
        # Write comments
        for comment in comments:
            f.write(comment)
        
        # Write data
        for idx, row in df.iterrows():
            line = ' '.join(row.values) + '\n'
            f.write(line)
    
    print(f"Coordinates swapped! Output saved to {output_file}")

# Usage
if __name__ == "__main__":

    input_file = r"C:\Users\samkr\Downloads\AA1460.swc"
    output_file = r"C:\Users\samkr\OneDrive\Documents\GitHub\Reconstruction_code\reconstructions\data\IRNPARN_cells\swc_swapped\AA1460_swapped.swc"
    swap_xz_coordinates(input_file, output_file)