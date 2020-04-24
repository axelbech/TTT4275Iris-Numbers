def importdata(textfile.txt, matrix[]):
    f = open(textfile.txt, "r")

    string = [""]

    #Read a line
    while(1):
        line = f.readline()

        if not line:
            break

        #Read character in current line
        while(1):
            char = line.read(1)
            if char != ",":
                string.append(char)
            else if char == ",":
                for i in string:

            if not char:
                break
            //Append char to matrix


