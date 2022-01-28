from termcolor import colored

def record_training_data(training_data,running_score,running_total_loss,running_policy_loss,running_value_loss):
    '''
    :param training_data:
    :param average_term:
    :param average_total_loss:
    :param avearge_policy_loss:
    :param average_value_loss:
    :return:
    '''
    average_term = training_data['average_term']
    average_score = running_score / average_term
    average_total_loss = running_total_loss / average_term
    avearge_policy_loss = running_policy_loss / average_term
    average_value_loss = running_value_loss / average_term

    scores = average_score
    total_loss = average_total_loss
    policy_loss = avearge_policy_loss
    value_loss = average_value_loss

    training_data['average_term'] = average_term
    training_data['train_scores'].append(scores)
    training_data['train_total_losses'].append(total_loss)
    training_data['train_policy_losses'].append(policy_loss)
    training_data['train_value_losses'].append(value_loss)

def report_training_data(episode,training_data):
    '''
    :param training_data: dictionary
    :return:
    '''
    print(colored('-'*20,'yellow'))
    print(colored(f"Episode : {episode}  (Averaged over {training_data['average_term']} episodes)",'yellow'))
    print(colored(f"Score : {training_data['train_scores'][-1]:.5f}", 'cyan'))
    print(colored(f"Total Loss : {training_data['train_total_losses'][-1]:.5f}", 'cyan'))
    print(colored(f"Policy Loss : {training_data['train_policy_losses'][-1]:.5f}", 'cyan'))
    print(colored(f"Value Loss : {training_data['train_value_losses'][-1]:.5f}", 'cyan'))
    print()
