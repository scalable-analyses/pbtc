    .text
    .type add_values, %function
    .global add_values
 add_values:
    stp fp, lr, [sp, #-16]!
    mov fp, sp

    ldr w3, [x0]
    ldr w4, [x1]
    add w5, w3, w4
    str w5, [x2]

    ldp fp, lr, [sp], #16

    ret