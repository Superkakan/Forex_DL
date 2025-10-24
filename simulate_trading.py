
# Start with using some form of trading strategy and test while validating the model
# Direction function

## if lstm predict higher price for the next timestamp
## and if the news are positive, then buy etc

# under construction
#convert_to_trinary = False
#if (convert_to_trinary == True):
#    prediction_direction(preds,targets)

#convert to direction
def prediction_direction(preds,targets,treshold = 0.0010):
    direction_val = []
    direction_tar = []
    up = 1
    neutral = 0
    down = -1
    for i in range(len(preds)):
        #prediction var
        if i > 0: # these basically convert the prediction into an classification instead of regression
            if (preds[i] - preds[i-1]) > treshold: 
                direction_val.append(up)
            elif (preds[i] - preds[i-1]) < -treshold: # downward
                direction_val.append(down)
            else: #neutral
                direction_val.append(neutral)
            
            #targets
            if (targets[i] - targets[i-1]) > treshold: 
                direction_tar.append(up)
            elif (targets[i] - targets[i-1]) < -treshold: # downward
                direction_tar.append(down)
            else: #neutral
                direction_tar.append(neutral)
            
            if preds[i] > preds[i-1]:
                pass #up
            if preds [i] > targets[i]:
                pass # preds bigger than target
    correct_dir_pred = []
    correct_dir_pred_amount = 0
    #compare the results from the classification
    for i in range(len(direction_tar)):
        if direction_tar[i] == direction_val[i]:
            correct_dir_pred.append(i) # would be nice to store the prediction with the index
            correct_dir_pred_amount += 1

