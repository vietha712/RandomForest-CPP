#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void
print_pattern (int * pattern, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	printf ("%d ", pattern[i]);
    }
    printf ("\n");
}

static void
shuffle (int * pattern, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	pattern[i] = i;
    }
    print_pattern (pattern, n);
    for (i = n - 1; i > 0; i--) {
	int j;
	j = random () % (i+1);
	if (j != i) {
	    int swap;
	    swap = pattern[j];
	    pattern[j] = pattern[i];
	    pattern[i] = swap;
	}
    }
    print_pattern (pattern, n);
}

#define SIZE 20

int main ()
{
    int pattern[SIZE];
    srandom (time (0));
    shuffle (pattern, SIZE);
    return 0;
}