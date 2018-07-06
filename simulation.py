# taken from 
# https://fcpython.com/blog/much-cost-fill-panini-world-cup-album-simulations-python
import numpy as np

def simulation():
	stickersNeeded = 682
	packetsBought = 0
	stickersGot = []
	swapStickers = 0
	while stickersNeeded > 0:
	    
	    #Buy a new packet
	    packetsBought += 1

	    #For each sticker, do some things 
	    for i in range(0,5):
	        
	        #Assign the sticker a random number
	        stickerNumber = np.random.randint(0,682)

	        #Check if we have the sticker
	        if stickerNumber not in stickersGot:
	            
	            #Add it to the album, then reduce our stickers needed count
	            stickersGot.append(stickerNumber)
	            stickersNeeded -= 1

	        #Throw it into the swaps pile
	        else:
	            swapStickers += 1

	    # print(stickersNeeded)
	return packetsBought,swapStickers

packets_total = 0

for i in range(1,100):
	packetsBought,swapStickers = simulation()
	print("Simulation #{}: Packets needed:{} Double (swap) stickers:{}".format(i,packetsBought,swapStickers))
	packets_total+=packetsBought


mean = packets_total/100
print("Mean packets needed: {}".format(mean))