/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License,
 *
 * Contributors:
 *     Hao Jiang - initial API and implementation
 *
 */

package org.apache.parquet.schema;

import org.apache.parquet.column.ColumnDescriptor;
import org.apache.parquet.column.Encoding;
import org.apache.parquet.io.InvalidRecordException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EncMessageType extends GroupType {

    private List<Encoding> encodings;

    private List<Object[]> params;


    private List<String[]> paths;

    private Map<String[], Integer> fieldIndexs = new HashMap<String[], Integer>();

    /**
     * @param name   the name of the type
     * @param fields the fields contained by this message
     */
    public EncMessageType(String name, List<Type> fields, List<Encoding> encodings, List<Object[]> params) {
        super(Repetition.REPEATED, name, fields);
        this.encodings = encodings;
        this.params = params;

        this.paths = getPaths();
        for (int i = 0; i < paths.size(); i++) {
            fieldIndexs.put(paths.get(i), i);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void writeToStringBuilder(StringBuilder sb, String indent) {
        sb.append("encmessage ")
                .append(getName())
                .append(getOriginalType() == null ? "" : " (" + getOriginalType() + ")")
                .append(" {\n");
        membersDisplayString(sb, "  ");
        sb.append("}\n");
    }

    /**
     * @return the max repetition level that might be needed to encode the
     * type at 'path'.
     */
    public int getMaxRepetitionLevel(String... path) {
        return getMaxRepetitionLevel(path, 0) - 1;
    }

    /**
     * @return the max repetition level that might be needed to encode the
     * type at 'path'.
     */
    public int getMaxDefinitionLevel(String... path) {
        return getMaxDefinitionLevel(path, 0) - 1;
    }

    protected int getFieldIndex(String[] path) {
        return fieldIndexs.get(path);
    }

    public Type getType(String... path) {
        return getType(path, 0);
    }

    public ColumnDescriptor getColumnDescription(String[] path) {
        int maxRep = getMaxRepetitionLevel(path);
        int maxDef = getMaxDefinitionLevel(path);
        PrimitiveType type = getType(path).asPrimitiveType();
        int fieldIndex = getFieldIndex(path);
        return new EncColumnDescriptor(path, type.getPrimitiveTypeName(),
                type.getTypeLength(), maxRep, maxDef, encodings.get(fieldIndex), params.get(fieldIndex));
    }

    public List<String[]> getPaths() {
        return paths;
    }

    public List<ColumnDescriptor> getColumns() {
        List<String[]> paths = this.getPaths();
        List<ColumnDescriptor> columns = new ArrayList<ColumnDescriptor>(paths.size());
        for (String[] path : paths) {
            columns.add(getColumnDescription(path));
        }
        return columns;
    }

    public boolean containsPath(String[] path) {
        return fieldIndexs.containsKey(path);
    }

    public MessageType asMessageType() {
        return null;
    }

}
