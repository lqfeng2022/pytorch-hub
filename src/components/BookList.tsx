import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
  List,
  ListItem,
  Text,
  HStack,
} from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import chapterList from '../data/chapterList';

const BookList = () => {
  const [ frontMatter ] = chapterList.slice(0, 1)
  const lists = chapterList.slice(1, 12)
  const backMatter = chapterList.slice(-3)

  return (
    <Accordion allowToggle >
      <List spacing={3} p='8px 16px'>
        <ListItem>
          <Link to={frontMatter.link}>
            <Text fontSize='md' _hover={{color: 'tomato'}}>{frontMatter.name}</Text>
          </Link>
        </ListItem>
      </List>
      {lists.map((list) => (
        <AccordionItem key={list.id}>
          <AccordionButton>
            <Box flex='1' textAlign='left'>
              <Link to={list.link!}>
                <HStack spacing={3} _hover={{color: 'tomato'}}>
                  <Text>{list.id}</Text>
                  <Text fontSize='md'>{list.name}</Text>
                </HStack>
              </Link>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            <List spacing={3}>
              {list.items?.map((item) => (
                <ListItem pl='22px' key={item.id}>
                  <Text fontSize='sm'>{item.name}</Text>
                </ListItem>
              ))}
            </List>
          </AccordionPanel>
        </AccordionItem>
      ))}
        <List spacing={3} p='8px 16px'>
          {backMatter.map((m) => <ListItem key={m.id}>
            <Link to={m.link}>
              <Text fontSize='md' _hover={{color: 'tomato'}}>{m.name}</Text>
            </Link>
          </ListItem>)}
        </List>
    </Accordion>
  );
};

export default BookList;