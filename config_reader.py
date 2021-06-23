def read_config_file(filename):
    file = open(filename, "r")
    vars = {}
    for line in file:
        line = line.rstrip('\n')
        var = line.split('=')
        variable_name = var[0].rstrip().lstrip()
        variable_value = var[1].rstrip().lstrip()
        if '[' in variable_value and ']' in variable_value:
            variable_value = variable_value.strip('[').strip(']')
            variable_value = variable_value.split(',')
            variable_value = [int(val) if val.isdigit() else val for val in variable_value]
            vars.update({variable_name: variable_value})
        else:
            vars.update({variable_name: int(variable_value) if variable_value.isdigit() else variable_value})
    return vars
