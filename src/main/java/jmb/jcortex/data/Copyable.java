/*
 * James Brundege
 * Date: 2017-04-06
 * MIT license: https://opensource.org/licenses/MIT
 */
package jmb.jcortex.data;

/** A copy method guaranteed to make a deep copy such that changes to the copy will not affect the original. */
public interface Copyable<T> {
    T copy();
}
