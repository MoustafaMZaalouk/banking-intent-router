# Error analysis: top confusions (tfidf model)

## virtual_card_not_working -> getting_virtual_card (13x)

- "Why is my virtual card is being declined?"
- "Can I make multiple online transactions with my virtual card?"

## virtual_card_not_working -> get_disposable_virtual_card (9x)

- "Why is my disposable virtual card being denied?"
- "My disposable virtual card is broken."

## card_acceptance -> card_not_working (8x)

- "What stores can I sue my card?"
- "I am traveling to Germany, Will I be able to use my card there?"

## why_verify_identity -> verify_my_identity (6x)

- "Do I really need to verify my identity?"
- "I won't verify my identity."

## unable_to_verify_identity -> verify_my_identity (5x)

- "I cannot verify my identity"
- "I can't verify my identity."

## request_refund -> Refund_not_showing_up (5x)

- "I need to do a refund"
- "Can I have a refund?"

## pin_blocked -> get_physical_card (4x)

- "Where can I view my PIN?"
- "Will you reinstate my PIN?"

## top_up_reverted -> top_up_failed (4x)

- "My top-up showed as complete, but it's no longer there! What's going on here?"
- "The app wouldn't accept my top up."

## card_payment_not_recognised -> extra_charge_on_statement (4x)

- "I think my statement has a fraudulent charge"
- "I have an unauthorized transaction on my statement"

## disposable_card_limits -> get_disposable_virtual_card (4x)

- "Are there any hindrances to using a disposable virtual card?"
- "What can I use a virtual disposable card for?"

# Error analysis: top confusions (embeddings model)

## declined_transfer -> declined_card_payment (6x)

- "Good morning. I tried to make a purchase with my credit card last night and again this morning. Both times it was declined. Can you investigate?"
- "How can I fix my card, it got declined twice."

## pending_transfer -> transfer_timing (5x)

- "How long can an EU transfer take?"
- "How long do transfer pend for?"

## get_disposable_virtual_card -> getting_virtual_card (5x)

- "how do VR cards work"
- "I would like a temporary virtual card"

## card_arrival -> card_delivery_estimate (4x)

- "When will I get my card?"
- "How long does a card delivery take?"

## wrong_amount_of_cash_received -> declined_cash_withdrawal (4x)

- "I don't know if this is an issue with the ATM or my account, but I just tried withdrawing 30 pounds from the ATM I'm at now and it only gave me 10. Is this a glitch or what exactly is going on?"
- "Why did I not get my cash back after I withdrew?"

## card_payment_not_recognised -> compromised_card (4x)

- "There are some strange charges on my card that I didn't make from a few days ago.  Is my card stolen and if so should I cancel it?  Will I get refunded the charges?"
- "Can you freeze my account?  I just saw there are transactions on my account that I don't recognize.  How can I fix this?"

## pending_transfer -> balance_not_updated_after_bank_transfer (4x)

- "How long for money transfer to show?"
- "Why is my transfer not done yet?"

## cash_withdrawal_not_recognised -> wrong_amount_of_cash_received (4x)

- "I didn't get cash from an ATM, but the app says I did"
- "I didn't get cash from an ATM, but according to the app a transaction was made when i didn't make it."

## extra_charge_on_statement -> card_payment_fee_charged (3x)

- "Where did this fee come from?"
- "There is an extra charge on my app that I was not told about or I was aware of!"

## pending_cash_withdrawal -> cash_withdrawal_not_recognised (3x)

- "I made a withdrawal from my account but it has not posted?"
- "I tried to take money from my card, but it didn't work. Later I saw that the transaction is still in progress. What's goign on?"

