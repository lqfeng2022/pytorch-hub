import Section from "./Section";

export default interface Chapter {
  id: number,
  name: string,
  description: string,
  sections: Section[],
}