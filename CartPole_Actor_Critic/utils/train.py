def record_training_data(training_data,average_term,running_total_loss,running_policy_loss,running_value_loss):
    '''
    :param training_data:
    :param average_term:
    :param average_total_loss:
    :param avearge_policy_loss:
    :param average_value_loss:
    :return:
    '''

    average_total_loss = running_total_loss / average_term
    avearge_policy_loss = running_policy_loss / average_term
    average_value_loss = running_value_loss / average_term
    total_loss = average_total_loss.to('cpu')
    policy_loss = avearge_policy_loss.to('cpu')
    value_loss = average_value_loss.to('cpu')

    training_data['average_term'] = average_term
    training_data['train_total_losses'].append(total_loss)
    training_data['train_policy_losses'].append(policy_loss)
    training_data['train_value_losses'].append(value_loss)
