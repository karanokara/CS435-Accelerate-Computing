/*
Measurement





Speedup

                ExecTime (origin)
    Speedup = ---------------------
                  ExecTime (new)


    e.g. 
    origin code in 10s, new code in 5s
    then
        speedup = 10s / 5s = 2

    

Amdahl's Law

                                                                    Fraction(enhanced)
    ExecTime(new) = ExecTime(orig) x [ (1 - Fraction(enhanced)) +  -------------------- ]
                                                                    Speedup(enhanced)

                  = "time in unchanged part" + "time in enhanced part"



               ExecTime (origin)
    Speedup = -------------------
                ExecTime (new)

                                ExecTime (origin)
            = -----------------------------------------------------------------------------
                                                                Fraction(enhanced)
                ExecTime(orig) x [ (1 - Fraction(enhanced)) +  -------------------- ]
                                                                 Speedup(enhanced)

                                    1
            = --------------------------------------------------
                                            Fraction(enhanced)
                (1 - Fraction(enhanced)) + --------------------
                                             Speedup(enhanced)





    Ex1.
        new processor is 10 times faster on computation 
        original processor is computation 40% of the time, and is I/O 60% of the time, 
        what is the overall speedup
            gained by incorporating the enhancement?

                                    1                                           1
            S = -------------------------------------------------- = ----------------------
                                            Fraction(enhanced)                   0.4
                (1 - Fraction(enhanced)) + --------------------     (1 - 0.4) + -----
                                             Speedup(enhanced)                   10

             = 1.56


    Ex2.
        FSQRT is responsible for 20%
        Improvement#1 speed up this operation by a factor of 10.
        Improvement#2 says, make FP instructions run faster by a factor of 1.6. 
        FP instructions are 50% of the execution time

                        1
        S1 = -------------------------- = 1.22
             ( 1 - 0.2 ) + (0.2 /10) 
            
                        1
        S2 = ---------------------------- = 1.23    (Faster)
             ( 1 - 0.5 ) + (0.5 /1.6) 




    Ex3. 
        enhancing machine by adding encryption hardware which is 20 times faster than orginal

        (a) What percentage of time must be spent in encryption in
            the original execution for adding encryption hardware to
            result in a speedup of 2?

                                    1                                    1
            S = -------------------------------------------------- = ----------------- = 2
                                            Fraction(enhanced)                  x
                (1 - Fraction(enhanced)) + --------------------      (1 - x) + ----
                                             Speedup(enhanced)                  20

            x = 0.53 = 53% of time in original


        (b) What percentage of time in the new execution will be
            spent on encryption operations if a speedup of 2 is
            achieved?

            origin time is 100
            then new time is 50
            other code takes (1 - 53%) = 47% in orgin time, so is 100 x 47% = 47
            so it takes 47 in new time, left 50 - 47 = 3 for encryption operations
            so % of time spent on encryption = 3 / 50 = 6%


        (c) Imagine that in the original program, 90% of the
            encryption operations could be done in parallel. What is
            the speedup of providing 2 or 4 encryption units?

            Assume other time is 47%
                encryption time is 53%
                    non-parallel: 53% x 10% = 5.3%
                    parallel: 53% x 90% = 47.7%

            providing 2 units:
                Total time become = 47% + (47.7% / 2) + 5.3% = 76.15%
                            100%
                speedup = --------- = 1.31
                            76.15%


            providing 4 units:
                Total time become = 47% + (47.7% / 4) + 5.3% = 64.23%
                            100%
                speedup = --------- = 1.56
                            64.23%








*/