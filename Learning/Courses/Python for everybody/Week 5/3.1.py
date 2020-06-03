def computepay(hours,rate) :
    hours = float(hours)
    rate = float(rate)
    if hours > 40.0 :
        return (((hours-40)*rate*0.5)+(hours*rate))
    else :
        return(hours*rate)
        quit()

computepay(hours,rate)
hours = input("Please enter hours")
rate = input("Please enter rate")