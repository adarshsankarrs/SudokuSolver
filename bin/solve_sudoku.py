import numpy as np

def grid_ref(number):
    """function converts linear 0-81 item reference to (y,x) item reference"""
    grid_ref = (number//9, number%9)
    #to get the row we divide by nine and ignore the remainder
    #to get the column we divide by nine and only look at the remainder
    return grid_ref

def value(grid, number):
    """function gets value of given linear item"""
    g_r = grid_ref(number)
    value = grid[g_r]
    #just check a given grid for a given grid reference
    return value

def cell(grid, number):
    """function returns array describing the cell that a linear item is in"""
    g_r = grid_ref(number)
    cell_ref = (g_r[0]//3, g_r[1]//3)
    cell = grid[((cell_ref[0])*3):((cell_ref[0])*3)+3, ((cell_ref[1])*3):((cell_ref[1])*3)+3]
    #cells are 3 wide and high
    return cell

def row(grid, number):
    """function returns array describing the row that a linear item is in"""
    g_r = grid_ref(number)
    row_ref = g_r[0]
    row = grid[(row_ref):(row_ref+1), 0:9]
    return row

def column(grid, number):
    """function returns array describing the column that a linear item is in"""
    g_r = grid_ref(number)
    column_ref = g_r[1]
    column = grid[0:9, (column_ref):(column_ref+1)]
    return column

def create_grid(puzzle_str):
    # Deleting whitespaces and newlines (\n)
    lines = puzzle_str.replace(' ','').replace('\n','')
    # Converting it to a list of digits
    digits = list(map(int, lines))
    # Converting it to a 9x9 numpy array
    grid = np.array(digits).reshape(9,9)
    print(grid)
    return grid

def solve(grid):
    forwards = True
    i = 0
    grid_original = np.array(grid, copy=True)
    while i <9*9:
        if value(grid_original, i) == 0 and forwards:
            for a in range(1, 10):
                if a not in cell(grid, i) and a not in row(grid, i) and a not in column(grid, i):
                    grid[grid_ref(i)] = a
                    i += 1
                    break
                else:
                    if a == 9:
                        forwards = False
                        i -= 1 #goes back a cell
                        break
        elif value(grid_original, i) != 0 and forwards:
            i += 1
        elif value(grid_original, i) == 0 and not forwards:
            if grid[grid_ref(i)] == 9:
                grid[grid_ref(i)] = 0
                i -= 1
            else:
                for a in range(grid[grid_ref(i)]+1, 10):
                    if a not in cell(grid, i) and a not in row(grid, i) and a not in column(grid, i):
                        grid[grid_ref(i)] = a
                        forwards = True
                        i += 1
                        break
                    else:
                        if a == 9:
                            grid[grid_ref(i)] = 0
                            i -= 1
                            break
        elif value(grid_original, i) != 0 and not forwards:
            i -= 1
    return(grid)