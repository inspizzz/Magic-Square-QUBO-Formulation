import math

# the total number of predicted magic squares of order n where the index of the list is n-3
totals = [1, 880, 275305224, 1.775399e19, 3.79809e34, 5.2225e54, 7.8448e79, 2.4149e110]

def format_number_scientifically(number):
    # Handle zero explicitly
    if number == 0:
        return "0.0x10^0"

    # Convert to scientific notation string with 16 digits of precision
    scientific_str = "{:.15e}".format(number)
    a_b, c_str = scientific_str.split("e")
    a_b = a_b.replace('.', '')  # remove decimal point

    # Remove leading zeros
    a_b = a_b.lstrip('0')

    # The first digit is 'a', the rest is 'b'
    a = a_b[0]
    b = a_b[1:]

    # Remove trailing zeros from b
    b = b.rstrip('0')

    # Parse exponent as integer
    c = int(c_str)

    return f"{a}.{b}x10^{c}"

for i in range(0, len(totals)):
    print(f"{i+3} -> {format_number_scientifically(totals[i])} -> {format_number_scientifically(math.factorial((i+3)**2))} -> {totals[i]*8 / math.factorial((i+3)**2)}")