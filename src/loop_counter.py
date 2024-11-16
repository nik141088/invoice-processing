import pickle
from src.constants import PREV_LOOP_ITERATION_FILE


def loop_counter_stack_clear():
    """
    this function clears the stack and its key
    :param: None
    :return: None
    """
    curr_inv = ''
    curr_data = list()
    with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
        pickle.dump([curr_inv, curr_data], fp)


def loop_counter_stack_pop(inv):
    """
    If the key matches with stored key, then one element is popped out of stack.
    If key doesn't match then stack is cleared
    :param inv: invoice name
    :return: data popped from stack. None if key doesn't match
    """
    with open(PREV_LOOP_ITERATION_FILE, 'rb') as fp:
        curr_inv, curr_data = pickle.load(fp)
    if inv != '' and curr_inv == inv:
        # all good
        if len(curr_data) == 0:
            return None
        else:
            # save data after popping
            curr_inv = inv
            ret_data = curr_data[-1]
            curr_data = curr_data[:-1]
            with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)
            return ret_data
    else:
        # clear stack
        loop_counter_stack_clear()
        return None


def loop_counter_stack_push(inv, new_data):
    """
    If pushing for the first time, then the key and the data are used to create a new stack
    If pushing to an existing stck then supplied key must match with the existing key.
    If they don't then stack is first cleared and new key, data pair is stored as the first element
    :param inv: invoice name
    :param new_data: data to be pushed to stack
    :return: None
    """
    # read current data
    with open(PREV_LOOP_ITERATION_FILE, 'rb') as fp:
        curr_inv, curr_data = pickle.load(fp)
    # if pushing for the first time
    if curr_inv == '':
        curr_inv = inv
        curr_data = list()
        if curr_inv is not None and curr_inv != '':
            curr_data.append(new_data)
        else:
            curr_inv = ''
            curr_data = list()
        with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
            pickle.dump([curr_inv, curr_data], fp)
    else:
        # if pushing to an existing stack, then check for invoice name
        if curr_inv == inv:
            # all good
            curr_inv = inv
            curr_data.append(new_data)
            with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)
        else:
            # clear stack
            loop_counter_stack_clear()
            # now write with new invoice
            curr_inv = inv
            curr_data = list()
            curr_data.append(new_data)
            with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
                pickle.dump([curr_inv, curr_data], fp)
